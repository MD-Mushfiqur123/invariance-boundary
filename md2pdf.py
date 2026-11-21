import os, re, matplotlib
import fpdf

FDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
SRC = "/workspace/scin_kill_experiment/PAPER_FULL_DRAFT.md"
OUT = "/workspace/scin_kill_experiment/PAPER_FULL_DRAFT.pdf"

class PDF(fpdf.FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-14)
        self.set_font("dv", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"PAPER_FULL_DRAFT v2.1 — page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

pdf = PDF(format="A4")
pdf.add_font("dv", "", os.path.join(FDIR, "DejaVuSans.ttf"))
pdf.add_font("dv", "B", os.path.join(FDIR, "DejaVuSans-Bold.ttf"))
pdf.add_font("dv", "I", os.path.join(FDIR, "DejaVuSans-Oblique.ttf"))
pdf.add_font("dv", "BI", os.path.join(FDIR, "DejaVuSans-BoldOblique.ttf"))
pdf.add_font("mono", "", os.path.join(FDIR, "DejaVuSansMono.ttf"))
pdf.set_margins(20, 18, 20)
pdf.set_auto_page_break(True, margin=20)

lines = open(SRC, encoding="utf-8").read().splitlines()
EPW = pdf.epw

def write_inline(txt, size, style=""):
    # inline code and emphasis -> styled runs
    pdf.set_font("dv", "BI" if "B" in style and "I" in style else style, size)
    segs = re.split(r"(`[^`]+`|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*)", txt)
    for s in segs:
        if not s:
            continue
        if s.startswith("`") and s.endswith("`"):
            pdf.set_font("mono", "", size - 1)
            pdf.write(5, s[1:-1])
            pdf.set_font("dv", style, size)
        elif s.startswith("***"):
            pdf.set_font("dv", "BI", size); pdf.write(5, s[3:-3]); pdf.set_font("dv", style, size)
        elif s.startswith("**"):
            pdf.set_font("dv", style + "B" if "B" not in style else style, size)
            pdf.write(5, s[2:-2]); pdf.set_font("dv", style, size)
        elif s.startswith("*"):
            pdf.set_font("dv", style + "I" if "I" not in style else style, size)
            pdf.write(5, s[1:-1]); pdf.set_font("dv", style, size)
        else:
            pdf.write(5, s)
    pdf.ln(5)

n = 0
while n < len(lines):
    ln = lines[n].rstrip(); n += 1
    if not ln.strip():
        continue
    if ln.startswith("# ") and n <= 2:
        pdf.add_page()
        pdf.set_font("dv", "B", 16); pdf.multi_cell(EPW, 7.5, ln[2:])
        pdf.ln(2)
    elif ln.startswith("### "):
        pdf.ln(1.5); pdf.set_font("dv", "B", 12.5)
        pdf.multi_cell(EPW, 6.5, ln[4:]); pdf.ln(1)
    elif ln.startswith("## "):
        pdf.ln(3); pdf.set_font("dv", "B", 14)
        pdf.multi_cell(EPW, 7, ln[3:]); pdf.ln(1.5)
    elif ln.strip() == "---":
        pdf.set_draw_color(160, 160, 160)
        y = pdf.get_y(); pdf.line(20, y + 1.5, 20 + EPW, y + 1.5); pdf.ln(5); pdf.set_draw_color(0, 0, 0)
    elif ln.lstrip().startswith("|"):
        rows = []
        while n - 1 < len(lines) and lines[n - 1].lstrip().startswith("|"):
            rows.append([c.strip() for c in lines[n - 1].strip().strip("|").split("|")]); n += 1
        rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in r)]
        if not rows: continue
        ncol = max(len(r) for r in rows)
        pdf.set_font("dv", "B", 9)
        w = EPW / ncol
        for j, r in enumerate(rows):
            style = "B" if j == 0 else ""
            pdf.set_font("dv", style, 9)
            for c in range(ncol):
                cell = r[c] if c < len(r) else ""
                border = "TB" if j == 0 else ("TB" if j == len(rows) - 1 else "")
                pdf.cell(w, 5.5, " " + cell[:110], border=border)
            pdf.ln(5.5)
        pdf.ln(2.5)
    elif re.match(r"^\s*[-*] ", ln) or re.match(r"^\s*\d+\. ", ln):
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
        indent = 6 if m.group(1) else 2
        pdf.set_x(20 + indent)
        write_inline(m.group(3) if not m.group(2).startswith("-") else "•  " + m.group(3), 10)
    elif ln.startswith("    ") or ln.startswith("\t"):
        pdf.set_font("mono", "", 9)
        pdf.multi_cell(EPW, 4.6, ln); pdf.ln(1)
    else:
        write_inline(ln, 10)

pdf.output(OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
