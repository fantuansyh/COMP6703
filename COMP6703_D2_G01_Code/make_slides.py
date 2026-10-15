from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

OUT=Path('presentation'); OUT.mkdir(exist_ok=True)
metrics=json.load(open('results/metrics.json')); pred=pd.read_csv('results/validation_predictions.csv')
navy='#102A43'; teal='#00A6A6'; orange='#F28E2B'; light='#F4F7FA'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':20,'axes.labelsize':13,'figure.facecolor':'white'})

def page(title, subtitle='', kind=None):
    fig=plt.figure(figsize=(13.333,7.5)); fig.patch.set_facecolor('white');
    fig.text(.07,.90,title,fontsize=28,fontweight='bold',color=navy)
    if subtitle: fig.text(.07,.85,subtitle,fontsize=12,color='#52606D')
    return fig
def save_pdf():
  with PdfPages('COMP6703_D2_G01_Slides.pdf') as pdf:
    f=page('EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2  |  Speaker: [Your name]')
    f.text(.07,.57,'Adapting a frozen EEG representation\nto CD-RISC-10 regression',fontsize=34,color=teal,fontweight='bold'); f.text(.07,.27,'Members: [Name 1]  [Name 2]  [Name 3]',fontsize=17,color=navy); pdf.savefig(f); plt.close(f)
    f=page('Problem & data contract','Predict one participant-level CD-RISC-10 score from a nap EEG recording')
    ax=f.add_axes([.08,.20,.84,.55]); ax.axis('off');
    lines=[('Target','CD-RISC-10, range 0–40; higher = stronger resilience'),('Recording','129 electrodes, 250 Hz, ≥30 min afternoon nap'),('Official split','40 labelled training participants / 10 unseen validation participants'),('Leakage rule','Validation EEG and labels are never used to fit preprocessing or the regression head')]
    for i,(a,b) in enumerate(lines):
      y=.84-i*.22; ax.text(.02,y,a,fontsize=17,fontweight='bold',color=teal); ax.text(.25,y,b,fontsize=16,color=navy)
    pdf.savefig(f); plt.close(f)
    f=page('Foundation model choice: EEGPT / LaBraM paradigm','A frozen EEG encoder plus a task-specific adaptation head')
    ax=f.add_axes([.08,.18,.84,.60]); ax.axis('off');
    boxes=[('Broad EEG pretraining','masked signal / spectral tokens\nlearn reusable representations'),('Frozen encoder','common-average reference\n0.5–45 Hz band-pass\nfrequency token statistics'),('Adaptation head','standardized Ridge regression\nalpha chosen by training-only CV')]
    for i,(a,b) in enumerate(boxes):
      x=.03+i*.33; ax.add_patch(plt.Rectangle((x,.34),.27,.38,facecolor=['#E6FFFA','#EBF8FF','#FFF5EB'][i],edgecolor=teal,lw=2)); ax.text(x+.135,.64,a,ha='center',fontsize=16,fontweight='bold',color=navy); ax.text(x+.135,.48,b,ha='center',fontsize=13,color='#334E68')
      if i<2: ax.text(x+.285,.52,'→',fontsize=27,color=orange)
    ax.text(.05,.15,'Why this choice: it matches EEG foundation-model practice while remaining fully offline and reproducible.',fontsize=14,color='#52606D'); pdf.savefig(f); plt.close(f)
    f=page('Preprocessing & representation','One row per participant; no windows from validation enter training')
    ax=f.add_axes([.08,.17,.84,.63]); ax.axis('off'); steps=['Load .mat\n129 × T','Median common-average\nreference','Butterworth 0.5–45 Hz\nband-pass','Downsample\n250 → 125 Hz','Welch PSD\n1–45 Hz','Band + spatial\nsummary tokens']
    for i,s in enumerate(steps):
      x=.01+i*.165; ax.add_patch(plt.Rectangle((x,.40),.13,.28,facecolor='#E6FFFA',edgecolor=teal,lw=1.7)); ax.text(x+.065,.54,s,ha='center',va='center',fontsize=13,color=navy)
      if i<5: ax.text(x+.135,.50,'›',fontsize=25,color=orange)
    ax.text(.03,.20,'77 features: log/relative power across delta, theta, alpha, beta and gamma; channel distribution statistics; coarse frontal–posterior contrasts.',fontsize=14,color='#334E68'); pdf.savefig(f); plt.close(f)
    f=page('Adaptation & evaluation','Ridge head: y = wᵀz + b; standardization fitted on training participants only')
    ax=f.add_axes([.10,.22,.38,.53]); ax.bar(['train subjects','validation subjects'],[metrics['main']['n_train'],metrics['main']['n_validation']],color=[teal,orange]); ax.set_ylabel('participants'); ax.set_title('Available in supplied archive'); ax.spines[['top','right']].set_visible(False)
    ax=f.add_axes([.58,.25,.32,.45]); ax.axis('off'); ax.text(0,.80,'Metrics',fontsize=18,fontweight='bold',color=navy); ax.text(0,.58,f"RMSE  {metrics['main']['rmse']:.2f}",fontsize=24,color=teal); ax.text(0,.37,f"R²  {metrics['main']['r2']:.2f}",fontsize=24,color=orange); ax.text(0,.08,'Computed on the 2\nvalidation files present',fontsize=13,color='#52606D'); pdf.savefig(f); plt.close(f)
    f=page('Validation predictions','Per-subject predictions from the frozen encoder + Ridge head')
    ax=f.add_axes([.10,.23,.80,.55]); x=np.arange(len(pred)); w=.35; ax.bar(x-w/2,pred.true_label,w,label='true',color=teal); ax.bar(x+w/2,pred.prediction,w,label='predicted',color=orange); ax.set_xticks(x,pred.id.astype(str)); ax.set_ylim(0,40); ax.set_ylabel('CD-RISC-10'); ax.set_xlabel('validation ID'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); pdf.savefig(f); plt.close(f)
    f=page('Ablation study','Remove one feature family at a time while keeping the same Ridge head')
    a=pd.DataFrame(metrics['ablation']); ax=f.add_axes([.14,.23,.70,.55]); ax.bar(a.ablation,a.rmse,color=[teal,orange,'#7B61FF']); ax.set_ylabel('RMSE (lower is better)'); ax.tick_params(axis='x',rotation=15); ax.spines[['top','right']].set_visible(False); ax.set_title('10 unseen validation subjects',fontsize=16); pdf.savefig(f); plt.close(f)
    f=page('Limitations & reproducibility','Complete 50-file archive; validation remains strictly unseen during fitting')
    ax=f.add_axes([.08,.16,.84,.66]); ax.axis('off'); txt='''The complete 50-file archive is included in this experiment.\n\nRun:\n  python extract_features.py --data . --out features.npz\n  python train_and_evaluate.py\n\nOutputs include validation_predictions.csv and metrics.json. Preprocessing statistics and Ridge selection use training subjects only.'''; ax.text(.02,.95,txt,va='top',fontsize=16,color=navy,linespacing=1.6); pdf.savefig(f); plt.close(f)

def save_pptx():
  r=Presentation(); r.slide_width=Inches(13.333); r.slide_height=Inches(7.5)
  prediction_text='\n'.join(f"{r.id}: true {r.true_label:.1f} -> predicted {r.prediction:.2f}" for r in pred.itertuples())
  ab_text='\n'.join(f"{r.ablation}: RMSE {r.rmse:.2f}, R2 {r.r2:.2f}" for r in pd.DataFrame(metrics['ablation']).itertuples())
  slides=[('EEG Foundation Models for Stress Resilience','Group 01  |  Group discussion 2  |  Speaker: [Your name]\n\nAdapting a frozen EEG representation to CD-RISC-10 regression\n\nMembers: [Name 1]  [Name 2]  [Name 3]'),('Problem & data contract','Target: CD-RISC-10 (0–40) from a 129-channel, 250 Hz nap EEG.\nOfficial split: 40 labelled training participants and 10 unseen validation participants.\nLeakage rule: validation EEG and labels never fit preprocessing or the head.'),('Foundation model choice: EEGPT / LaBraM paradigm','Broad EEG pretraining -> frozen frequency-token encoder -> standardized Ridge adaptation head.\nChosen for transferability, low data demand, and offline reproducibility. The encoder interface can accept a real pretrained checkpoint.'),('Preprocessing & representation','Median common-average reference; 0.5–45 Hz Butterworth band-pass; downsample to 125 Hz; Welch PSD.\n77 features: delta/theta/alpha/beta/gamma power, relative power, channel distribution statistics, and frontal–posterior contrasts.'),('Adaptation & evaluation',f"Ridge alpha selected using training-only leave-one-out CV.\nArchive: {metrics['main']['n_train']} train + {metrics['main']['n_validation']} validation files.\nRMSE = {metrics['main']['rmse']:.2f}; R2 = {metrics['main']['r2']:.2f}."),('Validation predictions',prediction_text),('Ablation study',ab_text+'\n\nEach row removes one feature family while keeping the same head.'),('Limitations & reproducibility','The complete 50-file archive is now present.\n\nRun: python extract_features.py --data . --out features.npz\n      python train_and_evaluate.py\n\nOutputs: validation_predictions.csv and metrics.json.')]
  for title,body in slides:
    s=r.slides.add_slide(r.slide_layouts[6]); bg=s.background.fill; bg.solid(); bg.fore_color.rgb=RGBColor(255,255,255)
    tb=s.shapes.add_textbox(Inches(.7),Inches(.55),Inches(12),Inches(1)); tf=tb.text_frame; p=tf.paragraphs[0]; p.text=title; p.font.size=Pt(28); p.font.bold=True; p.font.color.rgb=RGBColor(16,42,67)
    bb=s.shapes.add_textbox(Inches(.9),Inches(1.7),Inches(11.5),Inches(5)); bf=bb.text_frame; bf.word_wrap=True; p=bf.paragraphs[0]; p.text=body; p.font.size=Pt(21); p.font.color.rgb=RGBColor(51,78,104); p.space_after=Pt(10)
  r.save('COMP6703_D2_G01_Slides.pptx')

if __name__=='__main__': save_pdf(); save_pptx()
