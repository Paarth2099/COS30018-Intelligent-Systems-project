"""Explicit desktop smoke test: real Tk callbacks and models, mocked dialogs.

Run separately; opens the app briefly and saves screenshots. Never selects or
uploads a user's files externally. Not part of ordinary unittest discovery.
"""
from pathlib import Path
from datetime import datetime
import json
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import Image,ImageGrab
from app import HNRSApp,ROOT

out=ROOT/'evidence/gui'/('smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True,exist_ok=False)
root=tk.Tk();app=HNRSApp(root);root.geometry('1180x860+30+30')
errors=[];report={}


def wait():
    deadline=time.monotonic()+90
    while app.busy and time.monotonic()<deadline:
        root.update();time.sleep(.04)
    root.update()
    assert not app.busy,'UI operation timed out'


def screenshot(name):
    root.deiconify();root.lift();root.attributes('-topmost',True)
    root.update();time.sleep(.3);root.update()
    x,y=root.winfo_rootx(),root.winfo_rooty()
    ImageGrab.grab(bbox=(x,y,x+root.winfo_width(),y+root.winfo_height())).save(out/name)
    root.attributes('-topmost',False)


try:
    with patch('app.messagebox.showerror',side_effect=lambda title,text,**kw:errors.append(text)):
        root.update()
        photo=ROOT/'data/segmentation_photos/Manula/Manula_07_label-420.jpg'
        with patch('app.filedialog.askopenfilename',return_value=str(photo)):
            app.load_button.invoke();wait()
        assert app.origin['route']=='file' and app.result is None
        app.recognise_button.invoke();wait()
        assert app.result['prediction']['text']=='420'
        report['loaded_photo_prediction']=app.prediction.get()
        screenshot('loaded_photo.png')
        with patch('app.filedialog.askdirectory',return_value=str(out)):
            app.export_button.invoke()
        assert len(list(out.glob('gui_*/result.json')))==1
        app.method.set('projection');app.settings_changed()
        app.recognise_button.invoke();wait()
        assert app.result['prediction']['text']=='47'
        report['projection_photo_prediction']=app.prediction.get()
        app.method.set('components');app.settings_changed()
        app.routes.select(1);app.number.set('007');app.generate_button.invoke();wait()
        assert app.origin['requested_text']=='007' and app.result is None
        app.recognise_button.invoke();wait()
        report['generated_request']='007';report['generated_prediction']=app.result['prediction']['text']
        assert len(app.result['segmentation'].boxes)==3
        screenshot('generated_number.png')
        root.geometry('1020x860');root.update()
        assert app.status_label.winfo_rooty()+app.status_label.winfo_height() <= root.winfo_rooty()+root.winfo_height()
        screenshot('minimum_window.png')
        root.geometry('1180x860');root.update()
        with patch('app.filedialog.askdirectory',return_value=str(out)):
            app.export_button.invoke()
        exported=[json.loads(p.read_text()) for p in out.glob('gui_*/result.json')]
        assert any(r['input'].get('requested_text')=='007' and r['prediction']['text']=='007' for r in exported)
        app.model.set('mlp');app.settings_changed()
        assert app.result is None and str(app.export_button['state'])=='disabled'
        app.recognise_button.invoke();wait()
        report['mlp_generated_prediction']=app.result['prediction']['text']
        assert app.result['prediction']['model']=='mlp'
        with patch('app.filedialog.askopenfilename',return_value=''):
            previous=app.image;app.load_button.invoke();assert app.image is previous
        app.number.set('abc');app.generate_button.invoke();wait()
        assert errors and '1 to 12 digits' in errors[-1];errors.clear()
        app.set_image(Image.new('RGB',(100,100),'white'),{'route':'file','file':'synthetic_blank'})
        app.recognise_button.invoke();wait()
        assert app.prediction.get()=='No digits detected'
        assert not errors,errors
        report['checks']='PASS: load, recognition, both exports, generation, leading-zero request, model and segmentation switching, stale-result clearing, cancel, invalid request, blank image'
        (out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        print('Evidence:',out,flush=True)
finally:
    if not app.closed:app.close()
