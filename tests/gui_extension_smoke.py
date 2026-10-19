"""Real Tk callbacks/models, mocked file dialogs; local screenshots and evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import Image, ImageGrab
from app import HNRSApp, ROOT

out=ROOT/'evidence/gui_extension'/('smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True)
root=tk.Tk();app=HNRSApp(root);root.geometry('1240x860+15+15')
errors=[];report={'created_utc':datetime.now(timezone.utc).isoformat(),'predictions':[],'checks':[]}
ticks=[0]


def heartbeat():
    ticks[0]+=1
    if not app.closed:root.after(20,heartbeat)


def wait():
    deadline=time.monotonic()+120
    while app.busy and time.monotonic()<deadline:
        root.update();time.sleep(.02)
    root.update();assert not app.busy,'GUI worker timeout'


def screenshot(name):
    root.lift();root.attributes('-topmost',True);root.update();time.sleep(.25)
    # Tk coordinates may be DPI-virtualised on Windows. Query physical client
    # coordinates for capture without changing the application's DPI context.
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        user32=ctypes.windll.user32
        user32.SetThreadDpiAwarenessContext.argtypes=[ctypes.c_void_p]
        user32.SetThreadDpiAwarenessContext.restype=ctypes.c_void_p
        previous=user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        try:
            rect=wintypes.RECT();point=wintypes.POINT(0,0)
            hwnd=wintypes.HWND(root.winfo_id())
            assert user32.GetClientRect(hwnd,ctypes.byref(rect))
            assert user32.ClientToScreen(hwnd,ctypes.byref(point))
            bounds=(point.x,point.y,point.x+rect.right,point.y+rect.bottom)
            ImageGrab.grab(bbox=bounds,all_screens=True).save(out/name)
            report.setdefault('captures',[]).append({'file':name,'physical_client_bounds':bounds,
                'tk_size':[root.winfo_width(),root.winfo_height()]})
        finally:
            user32.SetThreadDpiAwarenessContext(previous)
    else:
        x,y=root.winfo_rootx(),root.winfo_rooty()
        ImageGrab.grab(bbox=(x,y,x+root.winfo_width(),y+root.winfo_height())).save(out/name)
    root.attributes('-topmost',False)


def load(path,valid=True):
    with patch('app.filedialog.askopenfilename',return_value=str(path)):
        app.load_button.invoke();wait()
    assert app.result is None
    if valid:
        assert not errors,errors
        assert app.origin['file']==str(Path(path).resolve())


def recognise():
    before=ticks[0];app.recognise_button.invoke()
    assert app.busy and str(app.combos[0]['state'])=='disabled'
    wait();assert app.result is not None,errors
    assert ticks[0]>before,'Event loop did not advance during worker'
    report['predictions'].append({'mode':app.mode.get(),'model':app.model.get(),
                                 'source':app.source_text.get(),'text':app.prediction.get(),
                                 'count':len(app.result['segmentation'].boxes)})


def export():
    before=set(out.glob('gui_*'))
    with patch('app.filedialog.askdirectory',return_value=str(out)):app.export_button.invoke()
    added=set(out.glob('gui_*'))-before;assert len(added)==1
    destination=added.pop();record=json.loads((destination/'result.json').read_text())
    assert record['settings']['mode']==app.mode.get()
    assert record['prediction']['text']==app.result['prediction']['text']
    return destination,record


try:
    root.after(20,heartbeat)
    with patch('app.messagebox.showerror',side_effect=lambda title,text,**kw:errors.append(text)):
        root.update();app.mode.set('Alphanumeric');app.mode_changed()
        assert app.method.get()=='grouped_components'
        assert str(app.combos[3]['state'])=='disabled'
        photo=ROOT/'data/own_strings/development/Manula/Manula_02_text-Room7.jpg'
        load(photo);recognise()
        expected=json.loads((ROOT/'evidence/character_segmentation/run_04/Manula/Manula_02_text-Room7/grouped_components/prediction.json').read_text())
        assert app.result['prediction']['text']==expected['text']
        assert len(app.digit_photos)==5
        screenshot('alphanumeric_photo.png');destination,record=export()
        assert (destination/'detected_characters.png').is_file()
        app.method.set('projection');app.settings_changed()
        assert app.result is None and str(app.export_button['state'])=='disabled'
        recognise();assert len(app.result['segmentation'].boxes)==4
        app.method.set('grouped_components');app.settings_changed()
        app.model.set('mlp');app.settings_changed();recognise()
        assert app.result['prediction']['model']=='mlp'
        app.model.set('cnn');app.settings_changed()
        app.routes.select(1);app.number.set('00Ai');app.generate_button.invoke();wait()
        assert app.origin['requested_text']=='00Ai'
        assert [s['character'] for s in app.origin['sources']]==list('00Ai')
        recognise();assert len(app.result['segmentation'].boxes)==4
        screenshot('generated_characters.png');_,record=export()
        assert record['input']['requested_text']=='00Ai'
        root.geometry('1020x860');root.update();screenshot('minimum_window.png')
        sidebar=app.export_button.master
        assert app.export_button.winfo_rooty()+app.export_button.winfo_height()<=sidebar.winfo_rooty()+sidebar.winfo_height()
        assert app.status_label.winfo_rooty()+app.status_label.winfo_height()<=root.winfo_rooty()+root.winfo_height()
        root.geometry('1240x860');root.update()
        app.number.set('a b');app.generate_button.invoke();wait()
        assert '1 to 12' in errors.pop()
        app.number.set('C');app.generate_button.invoke();wait()
        assert 'No single-character photo for: C' in errors.pop()
        invalid=out/'invalid.png';invalid.write_text('not an image')
        load(invalid,valid=False);assert errors.pop()
        with patch('app.filedialog.askopenfilename',return_value=''):
            previous=app.image;app.load_button.invoke();assert app.image is previous
        app.set_image(Image.new('RGB',(100,100),'white'),{'route':'file','file':'blank'})
        recognise();assert app.prediction.get()=='No characters detected';export()
        app.mode.set('Numbers');app.mode_changed()
        assert app.result is None and app.method.get()=='components'
        assert str(app.combos[3]['state'])=='readonly'
        load(ROOT/'data/segmentation_photos/Manula/Manula_07_label-420.jpg')
        recognise();assert app.prediction.get()=='420';destination,_=export()
        assert (destination/'detected_digits.png').is_file()
        screenshot('numbers_regression.png')
        app.routes.select(1);app.number.set('007');app.generate_button.invoke();wait()
        recognise();assert app.prediction.get()=='007'
        app.model.set('mlp');app.settings_changed();recognise()
        assert app.result['prediction']['model']=='mlp'
        assert len(app.service.predictors)==4
        assert not errors,errors
        report['checks']=['Alphanumeric photo matches saved pipeline','Both character models',
                          'Both character segmentation methods','Character generation preserves case/zeros',
                          'Source request separate from actual prediction','Image/mode/model changes clear stale results',
                          'Invalid file and text, missing samples, cancelled dialog, blank image',
                          'Character/number/blank exports','Numbers photo and generated 007 regression',
                          'Four mode/model caches isolated','GUI heartbeat during recognition, controls disabled',
                          'Minimum window controls/status within bounds']
        report['status']='PASS'
        report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['app.py','gui_core.py','tests/gui_extension_smoke.py']}
        (out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True);print('Evidence:',out,flush=True)
except Exception as error:
    report['status']='FAIL';report['error']=repr(error);report['dialogs']=errors
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    raise
finally:
    if not app.closed:app.close()
