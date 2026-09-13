"""Build the public research PDF from its readable Markdown source."""
from pathlib import Path
import re
import html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
root=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('Lato',str(root/'assets/fonts/lato/Lato-Regular.ttf')))
pdfmetrics.registerFontFamily('Lato',normal='Lato',bold='Lato',italic='Lato',boldItalic='Lato')
ink=colors.HexColor('#172d3b');muted=colors.HexColor('#516b7b');accent=colors.HexColor('#b9532a')
styles={
 'body':ParagraphStyle('body',fontName='Lato',fontSize=10.2,leading=14.6,textColor=ink,spaceAfter=10),
 'h1':ParagraphStyle('h1',fontName='Lato',fontSize=26,leading=30,textColor=ink,spaceAfter=17),
 'h2':ParagraphStyle('h2',fontName='Lato',fontSize=15,leading=19,textColor=accent,spaceAfter=12),
 'cell':ParagraphStyle('cell',fontName='Lato',fontSize=9,leading=12,textColor=ink),
 'caption':ParagraphStyle('caption',fontName='Lato',fontSize=8,leading=11,textColor=muted,spaceAfter=10),
}
def inline(s):
 s=html.escape(s)
 s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<a href="{m[2]}" color="#a44c29">{m[1]}</a>',s)
 s=re.sub(r'`([^`]+)`',r'<font name="Courier" size="8.4">\1</font>',s)
 s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
 return s
story=[]
source=(root/'docs/research-report.md').read_text().replace('<!-- pagebreak -->','\n\n<!-- pagebreak -->\n\n')
blocks=re.split(r'\n\s*\n',source.strip())
for block in blocks:
 if block.strip()=='<!-- pagebreak -->':story.append(PageBreak());continue
 if block.startswith('!['):
  m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',block.strip());path=(root/'docs'/m[2]).resolve()
  w,h=PILImage.open(path).size;dw=520;dh=h/w*dw
  limit=60 if path.stem in ['adapter','dynamics','drive','readout'] else 34 if path.stem=='cer' else 245 if path.stem.endswith('-end') else dh
  if dh>limit: dw*=limit/dh;dh=limit
  story += [Image(str(path),width=dw,height=dh),Spacer(1,5),Paragraph(inline(m[1]),styles['caption'])];continue
 if block.startswith('|'):
  lines=block.splitlines();rows=[[Paragraph(inline(x.strip()),styles['cell']) for x in line.strip('|').split('|')] for line in lines if not re.fullmatch(r'[| :\-]+',line)]
  n=len(rows[0]);widths={2:[260,260],3:[250,135,135],4:[224,91,100,105]}[n]
  if lines[0].startswith('| Stage'): widths=[110,410]
  table=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=1)
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dde9ee')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f4f7f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9cb1bd'))]))
  story += [table,Spacer(1,13)];continue
 for line in block.splitlines() if block.startswith('#') else [block.replace('\n',' ')]:
  if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['h2']))
  elif line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['h1']))
  else:story.append(Paragraph(inline(line),styles['body']))
def frame(c,doc):
 c.setTitle('Fly OCR: A fixed fly circuit, a small trained decoder')
 c.setAuthor('Jerry Liu')
 c.setSubject('Methods, recorded results, limitations and reproducibility of a connectome OCR experiment')
 c.setFillColor(muted);c.setFont('Lato',8)
 c.drawString(46,766,'FLY / OCR     RESEARCH REPORT')
 c.drawRightString(566,766,'12 SEPTEMBER 2026')
 c.setStrokeColor(colors.HexColor('#c7d5dc'));c.line(46,754,566,754)
 c.drawString(46,28,'Fixed connectome. Engineered inputs. Measured predictions.')
 c.drawRightString(566,28,str(doc.page))
(root/'output/pdf').mkdir(parents=True,exist_ok=True)
target=root/'output/pdf/fly-ocr-research-report.pdf'
doc=SimpleDocTemplate(str(target),pagesize=(612,792),rightMargin=46,leftMargin=46,topMargin=55,bottomMargin=48)
doc.build(story,onFirstPage=frame,onLaterPages=frame)
(root/'docs/fly-ocr-research-report.pdf').write_bytes(target.read_bytes())
print(target)
