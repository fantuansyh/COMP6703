from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT=Path(__file__).parent; ASSET=ROOT/'presentation'; ASSET.mkdir(exist_ok=True)
M=json.load(open(ROOT/'results/metrics.json')); P=pd.read_csv(ROOT/'results/validation_predictions.csv')
RED='#A6192E'; DARK='#252525'; GRAY='#5B6770'; BLUE='#1F5A75'; GOLD='#C99700'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titleweight':'bold','axes.edgecolor':'#9AA3A8','axes.labelcolor':DARK,'xtick.color':DARK,'ytick.color':DARK})
def style(ax): ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color='#D9DEE2',lw=.7); ax.set_axisbelow(True)
def figures():
    fig,ax=plt.subplots(figsize=(11,5)); ax.axis('off'); vals=[[f'{int(x.id):03d}',f'{x.true_label:.1f}',f'{x.prediction:.2f}',f'{x.prediction-x.true_label:+.2f}'] for x in P.itertuples()]; t=ax.table(cellText=vals,colLabels=['ID','Observed','Predicted','Error'],loc='center',cellLoc='center',colWidths=[.18,.24,.24,.24]); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.65)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor('#D9DEE2'); cell.set_linewidth(.6); cell.set_facecolor(RED if r==0 else ('#F3F5F6' if r%2==0 else 'white')); cell.get_text().set_color('white' if r==0 else DARK); cell.get_text().set_weight('bold' if r==0 else 'normal')
    fig.tight_layout(); fig.savefig(ASSET/'validation_table.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,5.2)); ax.scatter(P.true_label,P.prediction,s=70,c=RED,edgecolors='white',lw=1.2); ax.plot([0,40],[0,40],'--',color=GRAY,lw=1.5,label='identity');
    for x in P.itertuples(): ax.annotate(f'{int(x.id):03d}',(x.true_label,x.prediction),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlim=(0,40),ylim=(0,45),xlabel='Observed CD-RISC-10',ylabel='Predicted CD-RISC-10',title='Unseen validation participants'); style(ax); ax.text(.04,.94,f"RMSE = {M['main']['rmse']:.2f}\nR² = {M['main']['r2']:.2f}",transform=ax.transAxes,va='top',fontsize=12,bbox=dict(facecolor='white',edgecolor='#D9DEE2')); fig.tight_layout(); fig.savefig(ASSET/'validation_scatter.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    a=pd.DataFrame(M['ablation']); fig,ax=plt.subplots(figsize=(7.2,4.7)); bars=ax.bar(['Full\nrepresentation','No relative\npower','Spectral\nonly'],a.rmse,color=[RED,BLUE,GOLD],width=.62); ax.set_ylabel('RMSE (lower is better)'); ax.set_title('Feature-family ablation on the same 10-person validation set'); style(ax)
    for b,v in zip(bars,a.rmse): ax.text(b.get_x()+b.get_width()/2,v+.35,f'{v:.2f}',ha='center',fontweight='bold'); fig.tight_layout(); fig.savefig(ASSET/'ablation.png',dpi=220,bbox_inches='tight'); plt.close(fig)
def pdf_page(pdf,title,kicker='',body=None,images=None):
    fig=plt.figure(figsize=(13.333,7.5),facecolor='white'); fig.text(.075,.925,'POLYU  |  COMP6703',fontsize=10,color=RED,weight='bold'); fig.text(.075,.855,title,fontsize=28,color=DARK,weight='bold');
    if kicker: fig.text(.075,.815,kicker,fontsize=12,color=GRAY)
    if body: fig.text(.075,.73,body,fontsize=16,color=DARK,va='top',linespacing=1.5)
    if images:
        for path,rect in images: ax=fig.add_axes(rect); ax.imshow(plt.imread(path)); ax.axis('off')
    pdf.savefig(fig); plt.close(fig)
def make_pdf():
    with PdfPages(ROOT/'COMP6703_D2_G01_Slides.pdf') as pdf:
        pdf_page(pdf,'EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2  |  Speaker: [Name]  |  Members: [Names]','Adapting a pretrained-style EEG representation to\nsubject-level CD-RISC-10 regression')
        pdf_page(pdf,'Study design','A strict participant-level split prevents information leakage',"Target  |  CD-RISC-10 score, range 0–40\nInput   |  129-channel afternoon-nap EEG at 250 Hz\nSplit   |  40 labelled training participants; 10 unseen validation participants\nRule    |  validation recordings and labels are never used to fit the adapter")
        pdf_page(pdf,'Model and adaptation','EEGPT family selected; offline frozen-encoder adapter reported','EEGPT / LaBraM-style broad EEG pretraining\n        ↓\nFrozen EEG representation: reference → band-pass → PSD tokens → spatial summaries\n        ↓\nTraining-only standardization + Ridge regression\n        ↓\nOne score per participant in [0, 40]\n\nCheckpoint discovery is included in the source package. Network access was unavailable in this run, so no external weights are claimed.')
        pdf_page(pdf,'Signal processing and features','A reproducible offline representation','1. Median common-average reference\n2. 0.5–45 Hz Butterworth band-pass\n3. Downsample 250 → 125 Hz\n4. Welch power spectral density\n5. Delta / theta / alpha / beta / gamma power, relative power and regional contrasts\n\nOutput: 77 features per participant; validation rows are transformed only after the encoder is fixed.')
        pdf_page(pdf,'Validation performance','All 10 validation participants are shown',images=[(ASSET/'validation_scatter.png',[.08,.18,.46,.58]),(ASSET/'validation_table.png',[.57,.20,.38,.48])])
        pdf_page(pdf,'Ablation study','One feature family removed at a time; the Ridge head and split are unchanged',images=[(ASSET/'ablation.png',[.16,.19,.68,.58])])
        pdf_page(pdf,'Interpretation and limitations','What the result says',f"The full representation obtains RMSE {M['main']['rmse']:.2f} and R² {M['main']['r2']:.2f}; the training-mean baseline is RMSE {M['main']['validation_mean_baseline_rmse']:.2f}.\nThe spectral-only ablation is {M['ablation'][2]['rmse']:.2f}; removing relative power is {M['ablation'][1]['rmse']:.2f}. Differences are small on n=10, so conclusions are directional.\n\nThe near-zero / negative R² indicates that this low-data adapter does not yet generalize beyond the validation-score mean. This is an honest baseline and motivates checkpoint-based transfer learning.")
        pdf_page(pdf,'Reproducibility','Source code, predictions and environment are included','conda activate codna\npython extract_features.py --data . --out features.npz\npython train_and_evaluate.py\npython make_slides_polished.py\n\nOutputs: results/validation_predictions.csv, results/metrics.json, presentation figures, PDF and PPTX.\nAlpha is selected by leave-one-subject-out CV on the 40 training participants only.')
def add_text(s,x,y,w,h,text,size=20,color=DARK,bold=False):
    b=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); p=b.text_frame.paragraphs[0]; p.text=text; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color.replace('#','')); return b
def make_pptx():
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5); predtext='\n'.join(f'{int(x.id):03d}: observed {x.true_label:.1f} -> predicted {x.prediction:.2f}' for x in P.itertuples()); abtext='\n'.join(f'{x.ablation}: RMSE {x.rmse:.2f}, R² {x.r2:.2f}' for x in pd.DataFrame(M['ablation']).itertuples())
    slides=[('EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2\nSpeaker: [Name]  |  Members: [Names]','text'),('Study design','Target: CD-RISC-10 (0–40)\nInput: 129-channel, 250 Hz afternoon-nap EEG\nSplit: 40 training / 10 unseen validation\nLeakage control: validation never fits the adapter','text'),('Model and adaptation','EEGPT family selected for broad EEG pretraining.\n\nOffline frozen-encoder adapter: band-power/topographic representation → training-only standardization → Ridge head.\nCheckpoint discovery is included; external weights were unavailable in this run.','text'),('Signal processing and features','Common-average reference → 0.5–45 Hz band-pass → 125 Hz → Welch PSD\n\n77 features: five canonical bands, relative power, channel distributions and regional contrasts.','text'),('Validation performance','', 'validation_scatter.png'),('Validation prediction table','', 'validation_table.png'),('Ablation study','', 'ablation.png'),('Interpretation and reproducibility',f"RMSE {M['main']['rmse']:.2f}  |  R² {M['main']['r2']:.2f}\nMean baseline RMSE {M['main']['validation_mean_baseline_rmse']:.2f}\n\nThe near-zero R² shows that this low-data adapter is a baseline, not a claim of strong generalization.\n\nconda activate codna\npython extract_features.py --data . --out features.npz\npython train_and_evaluate.py",'text')]
    for title,body,kind in slides:
        s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=RGBColor(255,255,255); add_text(s,.65,.35,12,.35,'POLYU  |  COMP6703',10,RED,True); add_text(s,.65,.78,12,.65,title,28,DARK,True)
        if kind=='text': add_text(s,.95,1.85,11.2,4.8,body,22,DARK)
        else: s.shapes.add_picture(str(ASSET/kind),Inches(1.1),Inches(1.55),width=Inches(11.1))
    prs.save(ROOT/'COMP6703_D2_G01_Slides.pptx')
if __name__=='__main__': figures(); make_pdf(); make_pptx()
