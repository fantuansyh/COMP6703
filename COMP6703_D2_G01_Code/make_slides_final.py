from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).parent; ASSET = ROOT / 'presentation'; ASSET.mkdir(exist_ok=True)
RED, DARK, GRAY, BLUE, GOLD = '#A6192E', '#252525', '#5B6770', '#1F5A75', '#C99700'
BIOT_ALL = json.load(open(ROOT/'results/biot_comparison.json'))
BIOT = next(x for x in BIOT_ALL if x['model'] == 'BIOT-six-datasets-18')
LAB = json.load(open(ROOT/'results/labram_comparison.json'))
FIN = json.load(open(ROOT/'results/biot_finetune.json')) if (ROOT/'results/biot_finetune.json').exists() else [
    {'mode':'head','rmse':9.7178,'r2':-0.3669},
    {'mode':'last_block','rmse':10.3510,'r2':-0.5508},
    {'mode':'full','rmse':8.8905,'r2':-0.1440},
]
P = pd.DataFrame(BIOT['predictions']).rename(columns={'true':'observed'})
BASE = json.load(open(ROOT/'results/metrics.json'))['main']
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titleweight':'bold','axes.edgecolor':'#9AA3A8'})

def style(ax):
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', color='#D9DEE2', lw=.7); ax.set_axisbelow(True)

def figures():
    fig, ax = plt.subplots(figsize=(6.6,5.2)); ax.scatter(P.observed,P.prediction,s=75,c=RED,edgecolors='white',lw=1.2); ax.plot([0,40],[0,40],'--',color=GRAY,lw=1.5)
    for x in P.itertuples(): ax.annotate(str(x.id),(x.observed,x.prediction),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlim=(0,40),ylim=(0,45),xlabel='Observed CD-RISC-10',ylabel='Predicted CD-RISC-10',title='BIOT six-datasets | unseen validation')
    style(ax); ax.text(.04,.94,f"RMSE = {BIOT['rmse']:.2f}\nR² = {BIOT['r2']:.3f}",transform=ax.transAxes,va='top',fontsize=12,bbox=dict(facecolor='white',edgecolor='#D9DEE2')); fig.tight_layout(); fig.savefig(ASSET/'validation_scatter_final.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,5)); vals=[[str(x.id),f'{x.observed:.1f}',f'{x.prediction:.2f}',f'{x.prediction-x.observed:+.2f}'] for x in P.itertuples()]; t=ax.table(cellText=vals,colLabels=['ID','Observed','Predicted','Error'],loc='center',cellLoc='center',colWidths=[.18,.24,.24,.24]); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.65)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor('#D9DEE2'); cell.set_linewidth(.6); cell.set_facecolor(RED if r==0 else ('#F3F5F6' if r%2==0 else 'white')); cell.get_text().set_color('white' if r==0 else DARK); cell.get_text().set_weight('bold' if r==0 else 'normal')
    ax.axis('off'); fig.tight_layout(); fig.savefig(ASSET/'validation_table_final.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    models=['Handcrafted\nRidge','BIOT PREST\n16ch','BIOT SHHS+\nPREST 18ch','BIOT six\ndatasets 18ch','LaBraM\nbase']; rmse=[BASE['rmse'],8.4713,8.7769,BIOT['rmse'],LAB['rmse']]
    fig,ax=plt.subplots(figsize=(9,4.8)); bars=ax.bar(models,rmse,color=[GRAY,BLUE,BLUE,RED,GOLD],width=.65); ax.set_ylabel('Validation RMSE (lower is better)'); ax.set_title('Frozen representation comparison'); style(ax)
    for b,v in zip(bars,rmse): ax.text(b.get_x()+b.get_width()/2,v+.12,f'{v:.2f}',ha='center',fontweight='bold'); fig.tight_layout(); fig.savefig(ASSET/'model_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    names=['Frozen BIOT\n+ Ridge','Head-only\nMLP','Last block\n+ head','Full\nfinetune','8-window\nfrozen']; ar=[BIOT['rmse']]+[x['rmse'] for x in FIN]+[8.9596]
    fig,ax=plt.subplots(figsize=(8.4,4.8)); bars=ax.bar(names,ar,color=[RED,BLUE,BLUE,BLUE,GOLD],width=.62); ax.set_ylabel('Validation RMSE (lower is better)'); ax.set_title('Adaptation ablation on BIOT six-datasets'); style(ax)
    for b,v in zip(bars,ar): ax.text(b.get_x()+b.get_width()/2,v+.15,f'{v:.2f}',ha='center',fontweight='bold'); fig.tight_layout(); fig.savefig(ASSET/'adaptation_ablation.png',dpi=220,bbox_inches='tight'); plt.close(fig)

def page(pdf,title,kicker='',body=None,images=None):
    fig=plt.figure(figsize=(13.333,7.5),facecolor='white'); fig.text(.075,.925,'POLYU  |  COMP6703',fontsize=10,color=RED,weight='bold'); fig.text(.075,.855,title,fontsize=28,color=DARK,weight='bold')
    if kicker: fig.text(.075,.815,kicker,fontsize=12,color=GRAY)
    if body: fig.text(.075,.73,body,fontsize=15,color=DARK,va='top',linespacing=1.45)
    if images:
        for path,rect in images: ax=fig.add_axes(rect); ax.imshow(plt.imread(path)); ax.axis('off')
    pdf.savefig(fig); plt.close(fig)

def make_pdf():
    with PdfPages(ROOT/'COMP6703_D2_G01_Slides.pdf') as pdf:
        page(pdf,'EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2  |  Speaker: [Name]  |  Members: [Names]','Adapting BIOT and LaBraM representations to subject-level CD-RISC-10 regression')
        page(pdf,'What is an EEG foundation model?','Pretrain broadly, adapt narrowly','An EEG foundation model learns reusable signal representations from many EEG datasets before seeing the downstream labels.\n\nAdaptation means freezing or lightly updating that representation and fitting a small task head. This is a good match for 40 labelled participants because it limits trainable degrees of freedom and reduces overfitting.')
        page(pdf,'Data and leakage control','The unit of splitting is the participant','Input  |  129-channel EGI nap EEG, 250 Hz\nTarget |  CD-RISC-10 score, range 0–40\nSplit  |  40 labelled training participants; 10 unseen validation participants\n\nAll windows from one participant stay in the same split. Validation labels are read once for the final score only; they never select a checkpoint, window rule, hyperparameter, or adapter.')
        page(pdf,'Models evaluated','Three public BIOT checkpoints and LaBraM base','BIOT | Biosignal Transformer, pretrained on MGH, SHHS and multi-dataset EEG. We compare three released checkpoints and select the six-datasets version.\n\nLaBraM | Large Brain Model, pretrained on roughly 2,500 hours across about 20 EEG datasets.\n\nBoth encoders are frozen during the main comparison; only a standardized Ridge head is fitted on training subjects.')
        page(pdf,'Input adaptation pipeline','Deterministic channel and sampling alignment','1. 0.5–45 Hz Butterworth band-pass for BIOT; 0.1–75 Hz for LaBraM\n2. Resample to the checkpoint contract: BIOT 200 Hz / 10 s; LaBraM 200 Hz / 8 s\n3. Pool all 128 EGI channels into 16, 18, or 62 deterministic contiguous regional channels\n4. Extract three evenly spaced windows per subject and average frozen embeddings\n5. Fit Ridge with training-only standardization and LOOCV alpha selection')
        page(pdf,'Frozen-model validation comparison','Same 40/10 split and Ridge protocol for every representation',images=[(ASSET/'model_comparison.png',[.12,.19,.76,.58])])
        page(pdf,'Best model on unseen validation','BIOT six-datasets, 18-channel checkpoint',images=[(ASSET/'validation_scatter_final.png',[.08,.18,.46,.58]),(ASSET/'validation_table_final.png',[.57,.20,.38,.48])])
        page(pdf,'Adaptation ablation','Updating more parameters hurts in this 40-subject regime',images=[(ASSET/'adaptation_ablation.png',[.15,.18,.70,.60])])
        page(pdf,'Findings and limitations','What we can defend','Best frozen result: RMSE %.2f, R² %.3f. This improves over handcrafted Ridge (RMSE %.2f) and LaBraM frozen probe (RMSE %.2f).\n\nHead-only, last-block and full fine-tuning were worse, indicating overfitting with 40 subjects. Eight windows also hurt; three evenly spaced windows are retained.\n\nThe channel mapping is deterministic regional pooling because electrode names were not supplied. Results are a strong, reproducible transfer baseline, not a claim of universal SOTA.' % (BIOT['rmse'],BIOT['r2'],BASE['rmse'],LAB['rmse']))
        page(pdf,'Reproducibility','All code and reported predictions are included','conda activate codna\npython evaluate_biot.py --models BIOT-six-datasets-18 --windows 3\npython evaluate_labram.py\npython train_and_evaluate.py\npython make_slides_final.py\n\nReported artifacts: results/biot_comparison.json, results/labram_comparison.json, results/biot_finetune.json and validation prediction tables.\nCheckpoint download URLs and loading notes are documented in README_SUBMISSION.md.')

def add_text(s,x,y,w,h,text,size=20,color=DARK,bold=False):
    b=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); p=b.text_frame.paragraphs[0]; p.text=text; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color.replace('#','')); return b

def make_pptx():
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    slides=[('EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2\nSpeaker: [Name]  |  Members: [Names]','text'),('What is an EEG foundation model?','Pretrain broadly on many EEG datasets, then adapt a frozen or lightly updated representation to a downstream task.\n\nThis is suitable for 40 labelled participants because the task head has far fewer degrees of freedom than the encoder.','text'),('Data and leakage control','129-channel EGI nap EEG at 250 Hz\n40 training participants / 10 unseen validation participants\n\nAll windows from one participant stay in one split. Validation labels are used only for the final metrics.','text'),('Models evaluated','BIOT: three released checkpoints (MGH, SHHS+MGH, six datasets)\nLaBraM: public base checkpoint pretrained on broad EEG\n\nMain comparison: frozen encoder + standardized Ridge head.','text'),('Input adaptation pipeline','Filter and resample to each model contract → deterministic regional channel pooling → three evenly spaced windows → frozen embeddings → Ridge head selected by training-only LOOCV.','text'),('Frozen-model comparison','', 'model_comparison.png'),('Best unseen validation','', 'validation_scatter_final.png'),('Adaptation ablation','', 'adaptation_ablation.png'),('Findings and reproducibility',f"BIOT six-datasets frozen: RMSE {BIOT['rmse']:.2f}, R² {BIOT['r2']:.3f}\nLaBraM frozen: RMSE {LAB['rmse']:.2f}, R² {LAB['r2']:.3f}\n\nFine-tuning overfits with 40 subjects; RL is not appropriate for static regression.\n\npython evaluate_biot.py --models BIOT-six-datasets-18 --windows 3\npython evaluate_labram.py",'text')]
    for title,body,kind in slides:
        s=prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb=RGBColor(255,255,255); add_text(s,.65,.35,12,.35,'POLYU  |  COMP6703',10,RED,True); add_text(s,.65,.78,12,.65,title,28,DARK,True)
        if kind=='text': add_text(s,.95,1.75,11.2,5.0,body,21,DARK)
        else: s.shapes.add_picture(str(ASSET/kind),Inches(1.1),Inches(1.55),width=Inches(11.1))
    prs.save(ROOT/'COMP6703_D2_G01_Slides.pptx')

if __name__=='__main__': figures(); make_pdf(); make_pptx()
