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

out=ROOT/'evidence/gui_week11'/('smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True)
root=tk.Tk();app=HNRSApp(root)
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
            user32.GetAncestor.argtypes=[wintypes.HWND,wintypes.UINT]
            user32.GetAncestor.restype=wintypes.HWND
            outer=wintypes.RECT();work=wintypes.RECT()
            assert user32.GetWindowRect(user32.GetAncestor(hwnd,2),ctypes.byref(outer))
            assert user32.SystemParametersInfoW(48,0,ctypes.byref(work),0)
            assert outer.left>=work.left and outer.top>=work.top and outer.right<=work.right and outer.bottom<=work.bottom, f'Outer window {[outer.left,outer.top,outer.right,outer.bottom]} exceeds physical work area {[work.left,work.top,work.right,work.bottom]}; Tk {root.geometry()}'
            ImageGrab.grab(bbox=bounds,all_screens=True).save(out/name)
            report.setdefault('captures',[]).append({'file':name,'physical_client_bounds':bounds,
                'physical_window_bounds':[outer.left,outer.top,outer.right,outer.bottom],
                'physical_work_area':[work.left,work.top,work.right,work.bottom],
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


def visible(widget):
    app.ensure_visible(widget);root.update()
    assert_visible(widget)


def assert_visible(widget):
    x,y=widget.winfo_rootx(),widget.winfo_rooty()
    vx,vy=app.viewport.winfo_rootx(),app.viewport.winfo_rooty()
    assert vx<=x and vy<=y
    assert x+widget.winfo_width()<=vx+app.viewport.winfo_width()
    assert y+widget.winfo_height()<=vy+app.viewport.winfo_height()


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
        assert app.model.get()=='resnet18'
        app.model.set('cnn');app.settings_changed()  # Preserve the historical regression checks below.
        assert app.method.get()=='grouped_components'
        assert str(app.combos[3]['state'])=='disabled'
        photo=ROOT/'data/own_strings/development/Manula/Manula_02_text-Room7.jpg'
        load(photo);recognise()
        regression=json.loads((ROOT/'tests/fixtures/gui_regression.json').read_text())
        assert app.result['prediction']['text']==regression['character_cnn_prediction']
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
        initial_geometry=root.geometry()
        root.geometry('800x500');root.update()
        visible(app.export_button);screenshot('small_window_controls.png')
        visible(app.digits_canvas);screenshot('small_window_crops.png')
        # FocusIn uses the same visibility path for keyboard navigation.
        app.folder_entry.focus_force();root.update();assert_visible(app.folder_entry)
        app.export_button.focus_force();root.update();assert_visible(app.export_button)
        app.viewport.yview_moveto(0);before=app.viewport.yview()
        app.viewport.event_generate('<MouseWheel>',delta=-120);root.update()
        assert app.viewport.yview()!=before,'Mouse wheel did not scroll'
        app.viewport.xview_moveto(0);before=app.viewport.xview()
        app.viewport.event_generate('<MouseWheel>',delta=-120,state=1);root.update()
        assert app.viewport.xview()!=before,'Shift-wheel did not scroll horizontally'
        root.geometry('640x400');root.update()
        visible(app.export_button);screenshot('minimum_window.png')
        sidebar=app.export_button.master
        assert app.export_button.winfo_rooty()+app.export_button.winfo_height()<=sidebar.winfo_rooty()+sidebar.winfo_height()
        assert app.status_label.winfo_rooty()+app.status_label.winfo_height()<=root.winfo_rooty()+root.winfo_height()
        root.geometry(initial_geometry);root.update();app.viewport.xview_moveto(0);app.viewport.yview_moveto(0)
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
        # Week 10 preprocessing result must reach both display and export.
        load(ROOT/'data/segmentation_photos/Saniru/Saniru_08_label-240.jpeg')
        recognise();assert app.prediction.get()=='240';_,record=export()
        assert record['prediction']['text']=='240'
        app.routes.select(1);app.number.set('007');app.generate_button.invoke();wait()
        recognise();assert app.prediction.get()=='007'
        app.model.set('mlp');app.settings_changed();recognise()
        assert app.result['prediction']['model']=='mlp'
        app.model.set('cnn');app.settings_changed()
        app.number.set('007007007007');app.generate_button.invoke();wait();recognise()
        assert app.origin['requested_text']=='007007007007' and len(app.digit_photos)==12
        app.digits_canvas.xview_moveto(1);root.update()
        assert app.digits_canvas.xview()[1]>.99
        visible(app.digits_canvas);screenshot('long_number_breakdown.png')
        _,record=export();assert record['prediction']['text']==app.result['prediction']['text']
        # Missing models are tested without changing or removing real checkpoints.
        from gui_core import RecognitionService
        real_service=app.service;app.service=RecognitionService(out/'missing_models')
        app.settings_changed();app.recognise_button.invoke();wait()
        assert errors.pop() and app.result is None and str(app.export_button['state'])=='disabled'
        app.service=real_service
        app.mode.set('Alphanumeric');app.mode_changed();app.model.set('cnn');app.settings_changed()
        for filename,text,count in [('Haresh_03_text-Hello.jpg','Hell0',5),('Haresh_06_text-Bill.jpg','Bill',4)]:
            load(ROOT/'data/own_strings/development/Haresh'/filename);recognise()
            assert app.prediction.get()==text and len(app.result['segmentation'].boxes)==count
            _,record=export();assert record['prediction']['text']==text
            app.viewport.yview_moveto(0);screenshot(filename.replace('.jpg','.png'))
        assert len(app.service.predictors)==4
        # Exercise every real checkpoint through Tk callbacks and exported settings.
        from gui_core import MODEL_NAMES,DEFAULT_MODELS
        assert tuple(app.combos[1]['values'])==MODEL_NAMES
        comparisons=regression['paired_photos']
        for mode,filename in [('Numbers','data/segmentation_photos/Manula/Manula_07_label-420.jpg'),
                              ('Alphanumeric','data/own_strings/development/Manula/Manula_02_text-Room7.jpg')]:
            app.mode.set(mode);app.mode_changed();assert app.model.get()==DEFAULT_MODELS[mode]
            assert app.result is None and str(app.export_button['state'])=='disabled'
            app.routes.select(0);load(ROOT/filename)
            expected=next(r for r in comparisons if r['file']==filename)
            for name in MODEL_NAMES:
                app.model.set(name);app.settings_changed()
                assert app.result is None and str(app.export_button['state'])=='disabled'
                recognise();assert app.result['prediction']['text']==expected[name]
                _,record=export()
                assert record['settings']['model']==name
                checkpoint=ROOT/record['settings']['checkpoint']
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==record['settings']['checkpoint_sha256']
                if name==DEFAULT_MODELS[mode]:
                    app.viewport.yview_moveto(0);screenshot('week11_'+mode.lower()+'_default.png')
            app.model.set('mobilenet_v3_small');app.settings_changed()
            app.viewport.yview_moveto(0);screenshot('week11_'+mode.lower()+'_model_selector.png')
        assert len(app.service.predictors)==8
        app.mode.set('Alphanumeric');app.mode_changed()
        app.routes.select(1);app.number.set('00Ai');app.generate_button.invoke();wait();recognise()
        assert app.model.get()=='resnet18' and len(app.digit_photos)==4
        _,record=export();assert record['input']['requested_text']=='00Ai'
        assert record['prediction']['text']==app.result['prediction']['text']
        app.viewport.yview_moveto(0);screenshot('week11_resnet_generated.png')
        # Missing and corrupt new models must surface errors, not substitute CNN.
        real_service=app.service
        app.service=RecognitionService(out/'missing_new_models')
        app.settings_changed();app.recognise_button.invoke();wait()
        assert errors.pop() and app.result is None and str(app.export_button['state'])=='disabled'
        corrupt_root=out/'corrupt_fixture';folder=corrupt_root/'experiments/week11_gpu_run01/emnist/resnet18'
        folder.mkdir(parents=True);(folder/'resnet18.pt').write_bytes(b'invalid checkpoint')
        (folder/'selection.json').write_text(json.dumps({'model':'resnet18','config':{'task':'emnist'},'checkpoint_sha256':'incorrect'}))
        app.service=RecognitionService(corrupt_root)
        app.settings_changed();app.recognise_button.invoke();wait()
        assert 'hash mismatch' in errors.pop() and app.result is None
        app.service=real_service;recognise()
        app.mode.set('Numbers');app.mode_changed();assert app.model.get()=='cnn'
        app.mode.set('Alphanumeric');app.mode_changed();assert app.model.get()=='resnet18'
        assert not errors,errors
        report['checks']=['Alphanumeric photo matches saved pipeline','Both character models',
                          'Both character segmentation methods','Character generation preserves case/zeros',
                          'Source request separate from actual prediction','Image/mode/model changes clear stale results',
                          'Invalid file and text, missing samples, cancelled dialog, blank image',
                          'Character/number/blank exports','Numbers photo and generated 007 regression',
                          'Eight mode/model caches isolated','GUI heartbeat during recognition, controls disabled',
                          'Physical outer window fits work area',
                          '800x500 and 640x400 scrolling; controls/crops reachable; fixed status',
                          'Mouse wheel and keyboard-focus visibility',
                          'Twelve-digit breakdown reaches final crop; missing checkpoint error recovers',
                          'Week 10 digit 240 and repaired Hello/Bill displayed and exported',
                          'All four architectures in both modes reproduce Week 11 saved photo predictions',
                          'Default models reset by mode; new-model exports include correct checkpoint hashes',
                          'ResNet generated characters; missing/corrupt new checkpoint errors and recovery']
        report['status']='PASS'
        import platform,torch,numpy,PIL
        report['environment']={'python':platform.python_version(),'platform':platform.platform(),
            'torch':torch.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,
            'tcl':root.tk.call('info','patchlevel'),'tk_screen':[root.winfo_screenwidth(),root.winfo_screenheight()],
            'executable':sys.executable,'native_dialogs':'mocked; not manually exercised',
            'clean_install':False,'other_computer':False}
        report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['app.py','gui_core.py','tests/gui_extension_smoke.py']}
        for name in report['source_sha256']:
            destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/name).read_bytes())
        (out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True);print('Evidence:',out,flush=True)
except Exception as error:
    report['status']='FAIL';report['error']=repr(error);report['dialogs']=errors
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    raise
finally:
    if not app.closed:app.close()
