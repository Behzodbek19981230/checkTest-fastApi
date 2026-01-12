import io
import re
import zipfile
import base64
import mimetypes
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from defusedxml.lxml import fromstring
from lxml import etree
from openpyxl import load_workbook


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NS = {
	"w": W_NS,
	"m": M_NS,
	"r": R_NS,
	"a": A_NS,
}


def _env_truthy(name: str, default: bool = False) -> bool:
	val = os.getenv(name)
	if val is None:
		return default
	return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _escape_html(text: str) -> str:
	return (
		str(text)
		.replace("&", "&amp;")
		.replace("<", "&lt;")
		.replace(">", "&gt;")
		.replace('"', "&quot;")
		.replace("'", "&#039;")
	)


def _strip_tags(html: str) -> str:
	return re.sub(r"<[^>]+>", " ", str(html or "")).strip()


def _is_filled(text_or_html: str) -> bool:
	s = str(text_or_html or "")
	if not s.strip():
		return False
	# Images are meaningful content in our DOCX HTML output.
	# Without this, cells that contain only <img> tags are treated as empty.
	if re.search(r"<\s*img\b", s, flags=re.I):
		return True
	# Latex marker or non-empty text
	if "$" in s:
		return True
	plain = _strip_tags(s)
	plain = re.sub(r"\s+", "", plain)
	return len(plain) > 0


def _omml_operator_to_latex(op: str) -> str:
	op = (op or "").strip()
	return {
		"∑": "\\sum",
		"∏": "\\prod",
		"∫": "\\int",
		"∮": "\\oint",
		"⋂": "\\bigcap",
		"⋃": "\\bigcup",
		"⋁": "\\bigvee",
		"⋀": "\\bigwedge",
	}.get(op, op)


def _first_child(el: etree._Element, local: str) -> Optional[etree._Element]:
	for c in el:
		if etree.QName(c).localname == local:
			return c
	return None


def _children(el: etree._Element, local: str) -> List[etree._Element]:
	return [c for c in el if etree.QName(c).localname == local]


def _clean_latex(s: str) -> str:
	s = (s or "").replace("\u00a0", " ")
	s = re.sub(r"\s+", " ", s).strip()
	# Common broken artifact from partial OMML nodes
	s = s.replace("\\frac{}{}", "")
	return s


def omml_to_latex(el: etree._Element) -> str:
	"""Best-effort OMML -> LaTeX.

	Covers the most common Word Equation constructs (fractions, scripts, radicals, n-ary, matrices).
	Unknown nodes fall back to concatenating children.
	"""
	name = etree.QName(el).localname

	# Text inside OMML runs
	if name == "t":
		return el.text or ""

	# Common wrappers
	if name in {"oMath", "oMathPara", "e", "r"}:
		return "".join(omml_to_latex(c) for c in el)

	if name == "f":
		num = _first_child(_first_child(el, "num") or el, "e")
		den = _first_child(_first_child(el, "den") or el, "e")
		num_latex = _clean_latex(omml_to_latex(num) if num is not None else "")
		den_latex = _clean_latex(omml_to_latex(den) if den is not None else "")
		if not num_latex and not den_latex:
			return ""
		# If one side is missing, keep MathJax-friendly tiny spacing placeholders.
		if not num_latex:
			num_latex = "\\,"
		if not den_latex:
			den_latex = "\\,"
		return f"\\frac{{{num_latex}}}{{{den_latex}}}"

	if name == "sSup":
		base = _first_child(el, "e")
		sup = _first_child(el, "sup")
		return f"{omml_to_latex(base) if base is not None else ''}^{{{omml_to_latex(sup) if sup is not None else ''}}}"

	if name == "sSub":
		base = _first_child(el, "e")
		sub = _first_child(el, "sub")
		return f"{omml_to_latex(base) if base is not None else ''}_{{{omml_to_latex(sub) if sub is not None else ''}}}"

	if name == "sSubSup":
		base = _first_child(el, "e")
		sub = _first_child(el, "sub")
		sup = _first_child(el, "sup")
		return (
			f"{omml_to_latex(base) if base is not None else ''}"
			f"_{{{omml_to_latex(sub) if sub is not None else ''}}}"
			f"^{{{omml_to_latex(sup) if sup is not None else ''}}}"
		)

	if name == "rad":
		e = _first_child(el, "e")
		deg = _first_child(el, "deg")
		deg_e = _first_child(deg, "e") if deg is not None else None
		if deg_e is not None and omml_to_latex(deg_e).strip():
			return f"\\sqrt[{omml_to_latex(deg_e)}]{{{omml_to_latex(e) if e is not None else ''}}}"
		return f"\\sqrt{{{omml_to_latex(e) if e is not None else ''}}}"

	if name == "nary":
		pr = _first_child(el, "naryPr")
		chr_el = pr.find("m:chr", namespaces=NS) if pr is not None else None
		chr_val = None
		if chr_el is not None:
			chr_val = chr_el.get(f"{{{W_NS}}}val") or chr_el.get("val")
		op = _omml_operator_to_latex(chr_val or "")

		sub = _first_child(el, "sub")
		sup = _first_child(el, "sup")
		e = _first_child(el, "e")

		out = op or "\\sum"
		if sub is not None and omml_to_latex(sub).strip():
			out += f"_{{{omml_to_latex(sub)}}}"
		if sup is not None and omml_to_latex(sup).strip():
			out += f"^{{{omml_to_latex(sup)}}}"
		body = omml_to_latex(e) if e is not None else ""
		if body:
			out += f"{{{body}}}"
		return out

	if name == "limLow":
		e = _first_child(el, "e")
		lim = _first_child(el, "lim")
		lim_e = _first_child(lim, "e") if lim is not None else None
		lim_s = omml_to_latex(lim_e) if lim_e is not None else ""
		return f"\\lim_{{{lim_s}}}{omml_to_latex(e) if e is not None else ''}"

	if name == "limUpp":
		e = _first_child(el, "e")
		lim = _first_child(el, "lim")
		lim_e = _first_child(lim, "e") if lim is not None else None
		lim_s = omml_to_latex(lim_e) if lim_e is not None else ""
		return f"\\lim^{{{lim_s}}}{omml_to_latex(e) if e is not None else ''}"

	if name == "d":
		# Delimiters (parentheses, brackets, etc.)
		pr = _first_child(el, "dPr")
		beg = pr.find("m:begChr", namespaces=NS) if pr is not None else None
		end = pr.find("m:endChr", namespaces=NS) if pr is not None else None
		beg_val = (beg.get(f"{{{W_NS}}}val") or beg.get("val")) if beg is not None else "("
		end_val = (end.get(f"{{{W_NS}}}val") or end.get("val")) if end is not None else ")"
		inner = "".join(omml_to_latex(c) for c in _children(el, "e"))
		return f"\\left{beg_val}{inner}\\right{end_val}"

	if name == "bar":
		e = _first_child(el, "e")
		return f"\\overline{{{omml_to_latex(e) if e is not None else ''}}}"

	if name == "m":
		# matrix: m:mr rows, each has m:e entries
		rows = []
		for mr in _children(el, "mr"):
			cells = [omml_to_latex(e) for e in _children(mr, "e")]
			rows.append(" & ".join(cells))
		body = " \\\\ ".join(rows)
		return f"\\begin{{matrix}}{body}\\end{{matrix}}"

	# Fallback: concat children
	return "".join(omml_to_latex(c) for c in el)


@dataclass
class ParsedQuestion:
	question: str
	options: List[str]
	correct_answer_index: int
	points: int


def _walk_tc_to_html(tc: etree._Element, image_by_rid: Optional[dict] = None) -> str:
	"""Extract cell contents (text + equations) into safe inline HTML.

	We output escaped text + `<br/>` + `$...$` LaTeX blocks. Frontend later turns LaTeX into SVG.
	"""
	parts: List[str] = []
	seen_image_rids = set()

	def walk(node: etree._Element):
		ns = etree.QName(node).namespace
		ln = etree.QName(node).localname

		# Embedded images inside Word tables (w:drawing -> a:blip r:embed="rIdX")
		if image_by_rid and ln == "blip":
			rid = node.get(f"{{{R_NS}}}embed")
			if rid and rid in image_by_rid:
				if rid in seen_image_rids:
					return
				seen_image_rids.add(rid)
				mime, b64 = image_by_rid[rid]
				parts.append(
					f'<img src="data:{mime};base64,{b64}" alt="image" '
					'style="max-width: 100%; height: auto; display: block; margin: 0.5em 0;" />'
				)
				return

		# Older Word exports can embed images via VML:
		#   <w:pict><v:shape>...<v:imagedata r:id="rIdX"/></v:shape></w:pict>
		if image_by_rid and ln == "imagedata":
			rid = node.get(f"{{{R_NS}}}id") or node.get(f"{{{R_NS}}}embed")
			if rid and rid in image_by_rid:
				if rid in seen_image_rids:
					return
				seen_image_rids.add(rid)
				mime, b64 = image_by_rid[rid]
				parts.append(
					f'<img src="data:{mime};base64,{b64}" alt="image" '
					'style="max-width: 100%; height: auto; display: block; margin: 0.5em 0;" />'
				)
				return

		# Word line breaks
		if ns == W_NS and ln in {"br", "cr"}:
			parts.append("<br/>")
			return

		# Word text
		if ns == W_NS and ln == "t":
			if node.text:
				parts.append(_escape_html(node.text))
			return

		# Word equations (OMML)
		if ns == M_NS and ln in {"oMath", "oMathPara"}:
			latex = omml_to_latex(node)
			latex = (latex or "").strip()
			latex = latex.replace("$", "")
			if latex:
				parts.append(f"${latex}$")
			return

		for child in list(node):
			walk(child)

	walk(tc)

	out = "".join(parts)
	out = re.sub(r"(<br\s*/?>\s*){3,}", "<br/><br/>", out, flags=re.I)
	return out.strip()


def _tc_colspan(tc: etree._Element) -> int:
	grid = tc.find("w:tcPr/w:gridSpan", namespaces=NS)
	if grid is None:
		return 1
	val = grid.get(f"{{{W_NS}}}val") or grid.get("val")
	try:
		return max(1, int(val or "1"))
	except Exception:
		return 1


def _extract_docx_tables(document_xml: bytes) -> List[etree._Element]:
	root = fromstring(document_xml)
	# Word doc body tables
	return root.findall(".//w:tbl", namespaces=NS)


def _extract_docx_table_rows(tbl: etree._Element, image_by_rid: Optional[dict] = None) -> List[List[str]]:
	rows: List[List[str]] = []
	prev_row: List[str] = []
	for tr in tbl.findall("w:tr", namespaces=NS):
		out_row: List[str] = []
		col = 0
		for tc in tr.findall("w:tc", namespaces=NS):
			html = _walk_tc_to_html(tc, image_by_rid=image_by_rid)

			# vertical merge support (simple carry-down)
			vmerge = tc.find("w:tcPr/w:vMerge", namespaces=NS)
			if vmerge is not None:
				val = vmerge.get(f"{{{W_NS}}}val") or vmerge.get("val") or "continue"
				if val != "restart" and col < len(prev_row) and prev_row[col]:
					html = prev_row[col]

			span = _tc_colspan(tc)
			# Ensure list length
			while len(out_row) <= col:
				out_row.append("")
			out_row[col] = html
			for s in range(1, span):
				while len(out_row) <= col + s:
					out_row.append("")
				if not out_row[col + s]:
					out_row[col + s] = ""
			col += span

		rows.append(out_row)
		prev_row = out_row
	return rows


def _score_table(rows: List[List[str]]) -> int:
	row_count = len(rows)
	max_cols = max((len(r) for r in rows), default=0)
	has_template_cols = 1 if max_cols >= 7 else 0
	return has_template_cols * 1_000_000 + max_cols * 1_000 + row_count


def parse_docx_questions(file_bytes: bytes) -> Tuple[List[ParsedQuestion], List[str]]:
	"""Parse DOCX template table into questions.

	Output question/option strings as HTML that may include `$...$` LaTeX markers.
	"""
	errors: List[str] = []
	questions: List[ParsedQuestion] = []
	lenient_correct = _env_truthy("IMPORT_LENIENT_CORRECT_ANSWER", default=False)

	with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
		xml = z.read("word/document.xml")
		# Build relationship map to resolve embedded images
		image_by_rid = {}
		try:
			rels_xml = z.read("word/_rels/document.xml.rels")
			rels_root = fromstring(rels_xml)
			for rel in rels_root.findall(".//{*}Relationship"):
				rid = rel.get("Id")
				target = rel.get("Target") or ""
				rtype = rel.get("Type") or ""
				if not rid or not target:
					continue
				# Only handle images
				if "relationships/image" not in rtype:
					continue
				# Normalize path to zip member
				norm = target.lstrip("/")
				if norm.startswith("../"):
					norm = norm.replace("../", "")
				if not norm.startswith("word/"):
					norm = f"word/{norm}"
				try:
					img_bytes = z.read(norm)
					ext = (norm.rsplit(".", 1)[-1] or "").lower()
					mime = mimetypes.types_map.get(f".{ext}", "application/octet-stream")
					b64 = base64.b64encode(img_bytes).decode("ascii")
					image_by_rid[rid] = (mime, b64)
				except Exception:
					continue
		except Exception:
			image_by_rid = {}

	tables = _extract_docx_tables(xml)
	if not tables:
		return [], ["DOCX ichida jadval topilmadi"]

	best_rows: Optional[List[List[str]]] = None
	best_score = -1
	for t in tables:
		rows = _extract_docx_table_rows(t, image_by_rid=image_by_rid)
		score = _score_table(rows)
		if score > best_score:
			best_score = score
			best_rows = rows

	rows = best_rows or []
	if not rows:
		return [], ["DOCX jadvalini o‘qib bo‘lmadi"]

	def _row_texts(row: List[str]) -> List[str]:
		return [_strip_tags(c or "").lower() for c in (row or [])]

	def _row_join(row: List[str]) -> str:
		return " ".join(_row_texts(row))

	def headerish(row: List[str]) -> bool:
		# Some templates have merged cells or shifted columns; be tolerant.
		texts = _row_texts(row)
		joined = " ".join(texts)
		has_savol = any("savol" in t for t in texts)
		has_togri = ("to'g'ri" in joined) or ("toʻgʻri" in joined) or ("tog'ri" in joined)
		has_ball = "ball" in joined
		has_variants = any(re.search(r"\b[abcd]\)?\b", t) for t in texts) or "a)" in joined or "b)" in joined
		return has_savol and (has_togri or has_ball) and (has_variants or has_ball)

	def titleish(row: List[str]) -> bool:
		# Title row often spans all columns and contains words like "shablon"/"import".
		joined = _row_join(row)
		filled_cells = sum(1 for t in _row_texts(row) if t.strip())
		return filled_cells <= 2 and ("shablon" in joined or "import" in joined)

	# Skip title/header rows (anything before the header row, plus header row itself)
	header_idx = next((i for i, r in enumerate(rows) if len(r) >= 5 and headerish(r)), -1)
	start_idx = header_idx + 1 if header_idx >= 0 else 0

	for i, row in enumerate(rows[start_idx:], start=start_idx):
		# Need at least 7 columns
		if len(row) < 7:
			continue
		if headerish(row) or titleish(row):
			continue

		q_html = row[0]
		opts_html = [row[1], row[2], row[3], row[4]]
		correct_raw = _strip_tags(row[5]).strip().upper()
		# Normalize common Cyrillic lookalikes users may paste from Word.
		correct_raw = correct_raw.translate(str.maketrans({
			"А": "A",
			"В": "B",
			"С": "C",
			"Д": "D",
		}))
		# Accept inputs like "A)", "A.", "javob: B", etc.
		m = re.search(r"\b([ABCD])\b", correct_raw) or re.search(r"([ABCD])", correct_raw)
		if m:
			correct_letter = m.group(1)
		else:
			# Also accept 1-4 as A-D
			m_num = re.search(r"\b([1-4])\b", correct_raw)
			if m_num:
				correct_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}[m_num.group(1)]
			else:
				correct_letter = ""
		points_raw = _strip_tags(row[6]).strip() or "1"
		try:
			points = int(float(points_raw))
		except Exception:
			points = 1

		if not _is_filled(q_html):
			errors.append(
				f"Qator {i + 1}: Savol matni bo‘sh (matn/rasm/formula topilmadi)"
			)
			continue
		if points < 1 or points > 10:
			errors.append(f"Qator {i + 1}: Ball 1-10 oralig‘ida bo‘lishi kerak")
			continue

		entries = []
		for letter, html in zip(["A", "B", "C", "D"], opts_html):
			if _is_filled(html):
				entries.append((letter, html))

		filled_letters = [l for (l, _) in entries]
		correct_letter_display = correct_letter or (correct_raw.strip() if correct_raw else "")

		if len(entries) < 2:
			errors.append(
				f"Qator {i + 1}: Kamida 2 ta variant bo‘lishi kerak. "
				f"To‘ldirilgan variantlar: {', '.join(filled_letters) if filled_letters else 'yo‘q'}."
			)
			continue

		idx = next((j for j, (letter, _) in enumerate(entries) if letter == correct_letter), -1)
		if idx < 0:
			if lenient_correct and correct_letter in {"A", "B", "C", "D"} and len(entries) > 0:
				# Map A/B/C/D to the 1st/2nd/3rd/4th FILLED option.
				# Example: filled = [A, C, D], correct=B -> choose 2nd filled => C.
				desired_pos = {"A": 0, "B": 1, "C": 2, "D": 3}[correct_letter]
				chosen = min(desired_pos, len(entries) - 1)
				errors.append(
					f"Qator {i + 1}: To‘g‘ri javob '{correct_letter_display}' bo‘lib, lekin o‘sha ustun bo‘sh. "
					f"Lenient rejim: {correct_letter} -> {filled_letters[chosen]} (to‘ldirilgan variantlar: {', '.join(filled_letters)})."
				)
				idx = chosen
			else:
				errors.append(
					f"Qator {i + 1}: To‘g‘ri javob A/B/C/D bo‘lishi va o‘sha variant to‘ldirilgan bo‘lishi kerak. "
					f"Keltirilgan: {correct_letter_display or 'bo‘sh'}. "
					f"To‘ldirilgan variantlar: {', '.join(filled_letters) if filled_letters else 'yo‘q'}."
				)
				continue

		questions.append(
			ParsedQuestion(
				question=q_html,
				options=[v for _, v in entries],
				correct_answer_index=idx,
				points=points,
			)
		)

	return questions, errors


def parse_xlsx_questions(file_bytes: bytes) -> Tuple[List[ParsedQuestion], List[str]]:
	errors: List[str] = []
	questions: List[ParsedQuestion] = []

	wb = load_workbook(filename=io.BytesIO(file_bytes), data_only=False)
	ws = wb.worksheets[0]

	# Expect header in first row; start from row 2
	for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
		values = ["" if v is None else str(v) for v in row]
		# ensure at least 7
		if len(values) < 7:
			continue
		q = values[0].strip()
		a, b, c, d = (values[1].strip(), values[2].strip(), values[3].strip(), values[4].strip())
		correct = values[5].strip().upper()
		points_raw = values[6].strip() or "1"
		try:
			points = int(float(points_raw))
		except Exception:
			points = 1

		if not q:
			errors.append(f"Qator {i}: Savol matni bo‘sh")
			continue
		if points < 1 or points > 10:
			errors.append(f"Qator {i}: Ball 1-10 oralig‘ida bo‘lishi kerak")
			continue

		entries = [("A", a), ("B", b), ("C", c), ("D", d)]
		entries = [(l, v) for (l, v) in entries if v]
		if len(entries) < 2:
			errors.append(f"Qator {i}: Kamida 2 ta variant bo‘lishi kerak")
			continue

		idx = next((j for j, (letter, _) in enumerate(entries) if letter == correct), -1)
		if idx < 0:
			errors.append(
				f"Qator {i}: To‘g‘ri javob A/B/C/D bo‘lishi va o‘sha variant to‘ldirilgan bo‘lishi kerak"
			)
			continue

		questions.append(
			ParsedQuestion(
				question=_escape_html(q),
				options=[_escape_html(v) for _, v in entries],
				correct_answer_index=idx,
				points=points,
			)
		)

	return questions, errors
