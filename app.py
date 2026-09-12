"""Saniru's local HNRS desktop GUI. Run with the project Python environment."""
import hashlib
import base64
from io import BytesIO
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image
from gui_core import open_photo, generate_number, annotated_image, export_result

ROOT=Path(__file__).resolve().parent


def tk_image(image,master):
    # Use Tk's PNG decoder: Pillow's optional _imagingtk DLL may fail in long
    # Windows checkout paths even when Pillow's regular image functions work.
    buffer=BytesIO();image.save(buffer,format='PNG')
    return tk.PhotoImage(master=master,data=base64.b64encode(buffer.getvalue()),format='png')


class HNRSApp:
    def __init__(self,root):
        self.root=root; self.image=None; self.origin=None; self.result=None
        self.settings=None; self.predictors={}; self.busy=False; self.closed=False
        self.messages=queue.Queue(); self.photos=[]; self.job_thread=None
        root.title('HNRS | Handwriting Studio'); root.geometry('1240x860'); root.minsize(1020,860)
        root.configure(bg='#f3f6fb')
        style=ttk.Style(root); style.theme_use('clam')
        style.configure('.',font=('Segoe UI',10))
        style.configure('TFrame',background='#f3f6fb')
        style.configure('TLabel',background='#f3f6fb',foreground='#192b46')
        style.configure('Panel.TFrame',background='white')
        style.configure('Panel.TLabel',background='white',foreground='#192b46')
        style.configure('Caption.TLabel',background='white',foreground='#64748b',font=('Segoe UI',9))
        style.configure('Section.TLabel',background='white',foreground='#334155',font=('Segoe UI',10,'bold'))
        style.configure('Title.TLabel',font=('Segoe UI',23,'bold'))
        style.configure('Muted.TLabel',foreground='#64748b')
        style.configure('TButton',padding=(12,9),background='#eef3fa',foreground='#26415e',borderwidth=0,font=('Segoe UI',10,'bold'))
        style.map('TButton',background=[('active','#dfe8f4')],foreground=[('disabled','#94a3b8')])
        style.configure('Accent.TButton',background='#2563eb',foreground='white',padding=(12,13))
        style.map('Accent.TButton',background=[('disabled','#dbe5f5'),('active','#1d4ed8')],foreground=[('disabled','#8192ab'),('!disabled','white')])
        style.configure('TEntry',padding=7,fieldbackground='#f8fafc',bordercolor='#dce4ef',lightcolor='#dce4ef',darkcolor='#dce4ef')
        style.configure('TCombobox',padding=6,fieldbackground='#f8fafc',background='#eef3fa',bordercolor='#dce4ef',arrowcolor='#475569')
        style.map('TCombobox',fieldbackground=[('readonly','#f8fafc')],selectbackground=[('readonly','#f8fafc')],selectforeground=[('readonly','#192b46')])
        style.configure('TNotebook',background='white',borderwidth=0)
        style.configure('TNotebook.Tab',padding=(12,8),background='#f1f5f9',foreground='#64748b')
        style.map('TNotebook.Tab',background=[('selected','#eaf1ff')],foreground=[('selected','#1d4ed8')])
        style.configure('Horizontal.TScrollbar',background='#dbe3ee',troughcolor='white',borderwidth=0,arrowsize=12)
        self.model=tk.StringVar(value='cnn'); self.method=tk.StringVar(value='components'); self.representation=tk.StringVar(value='otsu')
        self.folder=tk.StringVar(value=str(ROOT/'data/segmentation_photos'))
        self.number=tk.StringVar(value='007'); self.status=tk.StringVar(value='Load a photo or generate a number to begin.')
        self.source_text=tk.StringVar(value='No image selected')
        self.prediction=tk.StringVar(value='No prediction yet')
        self.details=tk.StringVar(value='Your recognition result will appear here.')

        frame=ttk.Frame(root,padding=24); frame.pack(fill='both',expand=True)
        header=ttk.Frame(frame);header.pack(fill='x',pady=(0,22))
        brand=tk.Label(header,text=' HN ',bg='#2563eb',fg='white',font=('Segoe UI',20,'bold'),padx=8,pady=8)
        brand.pack(side='left',padx=(0,14))
        titles=ttk.Frame(header);titles.pack(side='left')
        ttk.Label(titles,text='Handwriting Studio',style='Title.TLabel').pack(anchor='w')
        ttk.Label(titles,text='Turn handwritten digits into readable numbers.',style='Muted.TLabel').pack(anchor='w',pady=(2,0))
        tk.Label(header,text='COS30018  /  HNRS',bg='#e7eef9',fg='#43618a',font=('Segoe UI',9,'bold'),padx=12,pady=8).pack(side='right')

        body=ttk.Frame(frame);body.pack(fill='both',expand=True)
        body.columnconfigure(1,weight=1);body.rowconfigure(0,weight=1)
        sidebar=ttk.Frame(body,style='Panel.TFrame',padding=18,width=280)
        sidebar.grid(row=0,column=0,sticky='ns',padx=(0,20));sidebar.grid_propagate(False)
        sidebar.columnconfigure(0,weight=1)
        ttk.Label(sidebar,text='01  /  IMAGE SOURCE',style='Section.TLabel').grid(row=0,column=0,sticky='w',pady=(0,14))
        self.routes=ttk.Notebook(sidebar);self.routes.grid(row=1,column=0,sticky='ew')
        load_tab=ttk.Frame(self.routes,style='Panel.TFrame',padding=(0,14))
        gen_tab=ttk.Frame(self.routes,style='Panel.TFrame',padding=(0,14))
        self.routes.add(load_tab,text='Open photo');self.routes.add(gen_tab,text='Generate')
        ttk.Label(load_tab,text='Choose a handwriting photo',style='Section.TLabel').pack(anchor='w',pady=(0,8))
        ttk.Label(load_tab,text='Use one line of dark digits on pale paper.',style='Caption.TLabel',wraplength=232).pack(anchor='w',pady=(0,16))
        self.load_button=ttk.Button(load_tab,text='Choose image...',command=self.load_dialog);self.load_button.pack(fill='x')
        ttk.Label(load_tab,text='PNG, JPG, BMP or TIFF',style='Caption.TLabel').pack(anchor='w',pady=(10,0))
        ttk.Label(gen_tab,text='Single-digit photo folder',style='Caption.TLabel').pack(anchor='w')
        folder_row=ttk.Frame(gen_tab,style='Panel.TFrame');folder_row.pack(fill='x',pady=(5,9))
        self.folder_entry=ttk.Entry(folder_row,textvariable=self.folder,width=12);self.folder_entry.pack(side='left',fill='x',expand=True)
        self.folder_button=ttk.Button(folder_row,text='Browse',command=self.folder_dialog);self.folder_button.pack(side='right',padx=(5,0))
        ttk.Label(gen_tab,text='Number to create  ·  1–12 digits',style='Caption.TLabel').pack(anchor='w')
        self.number_entry=ttk.Entry(gen_tab,textvariable=self.number);self.number_entry.pack(fill='x',pady=(5,9))
        self.generate_button=ttk.Button(gen_tab,text='Generate image',command=self.generate);self.generate_button.pack(fill='x')
        ttk.Separator(sidebar).grid(row=2,column=0,sticky='ew',pady=(8,18))
        ttk.Label(sidebar,text='02  /  RECOGNITION',style='Section.TLabel').grid(row=3,column=0,sticky='w',pady=(0,12))
        options=ttk.Frame(sidebar,style='Panel.TFrame');options.grid(row=4,column=0,sticky='ew')
        self.combos=[]
        for label,variable,values in [('Recognition model',self.model,['cnn','mlp']),('Digit separation',self.method,['components','projection']),('Image preparation',self.representation,['otsu','fixed','grayscale'])]:
            ttk.Label(options,text=label,style='Caption.TLabel').pack(anchor='w',pady=(0,4))
            combo=ttk.Combobox(options,textvariable=variable,values=values,state='readonly',width=20)
            combo.pack(fill='x',pady=(0,10));combo.bind('<<ComboboxSelected>>',self.settings_changed);self.combos.append(combo)
        self.recognise_button=ttk.Button(sidebar,text='Recognise number',style='Accent.TButton',command=self.recognise)
        self.recognise_button.grid(row=5,column=0,sticky='ew',pady=(8,8))
        self.export_button=ttk.Button(sidebar,text='Export result...',command=self.export_dialog);self.export_button.grid(row=6,column=0,sticky='ew')

        workspace=ttk.Frame(body);workspace.grid(row=0,column=1,sticky='nsew')
        workspace.columnconfigure(0,weight=1);workspace.rowconfigure(1,weight=1)
        ttk.Label(workspace,textvariable=self.source_text,style='Muted.TLabel',wraplength=620).grid(row=0,column=0,sticky='w',pady=(0,10))
        panes=ttk.Frame(workspace);panes.grid(row=1,column=0,sticky='nsew')
        panes.columnconfigure(0,weight=1,uniform='panes');panes.columnconfigure(1,weight=1,uniform='panes');panes.rowconfigure(0,weight=1)
        for column,title,subtitle,attribute in [(0,'Original image','Your handwriting input','input_canvas'),(1,'Detected digits','Separated from left to right','output_canvas')]:
            card=ttk.Frame(panes,style='Panel.TFrame',padding=14);card.grid(row=0,column=column,sticky='nsew',padx=(0,7) if column==0 else (7,0))
            ttk.Label(card,text=title,style='Section.TLabel').pack(anchor='w')
            ttk.Label(card,text=subtitle,style='Caption.TLabel').pack(anchor='w',pady=(3,12))
            canvas=tk.Canvas(card,bg='#f8fafc',height=100,width=100,highlightthickness=0)
            canvas.pack(fill='both',expand=True);canvas.bind('<Configure>',lambda e:self.redraw());setattr(self,attribute,canvas)
        result_frame=tk.Frame(workspace,bg='#eaf1ff',padx=20,pady=14);result_frame.grid(row=2,column=0,sticky='ew',pady=(16,16))
        tk.Label(result_frame,text='RECOGNISED NUMBER',bg='#eaf1ff',fg='#43618a',font=('Segoe UI',9,'bold')).pack(anchor='w')
        tk.Label(result_frame,textvariable=self.prediction,bg='#eaf1ff',fg='#1747a6',font=('Segoe UI',29,'bold')).pack(anchor='w',pady=(3,2))
        tk.Label(result_frame,textvariable=self.details,bg='#eaf1ff',fg='#526b91',font=('Segoe UI',9),wraplength=600,justify='left').pack(anchor='w')
        digit_card=ttk.Frame(workspace,style='Panel.TFrame',padding=(14,12));digit_card.grid(row=3,column=0,sticky='ew')
        ttk.Label(digit_card,text='Digit breakdown',style='Section.TLabel').pack(anchor='w')
        ttk.Label(digit_card,text='Prepared crops and model scores · scores are not calibrated probabilities.',style='Caption.TLabel',wraplength=620).pack(anchor='w',pady=(3,8))
        self.digits_canvas=tk.Canvas(digit_card,bg='white',height=112,width=100,highlightthickness=0);self.digits_canvas.pack(fill='x')
        scroll=ttk.Scrollbar(digit_card,orient='horizontal',command=self.digits_canvas.xview);scroll.pack(fill='x');self.digits_canvas.configure(xscrollcommand=scroll.set)
        self.status_label=ttk.Label(frame,textvariable=self.status,style='Muted.TLabel',wraplength=1120)
        self.status_label.pack(anchor='w',pady=(14,0))
        frame.bind('<Configure>',lambda e:self.status_label.configure(wraplength=max(200,e.width-48)))
        self.controls=[self.load_button,self.folder_button,self.generate_button,self.folder_entry,self.number_entry]
        self.refresh_controls(); self.poll_id=root.after(80,self.poll); root.protocol('WM_DELETE_WINDOW',self.close)

    def refresh_controls(self):
        for widget in self.controls: widget.configure(state='disabled' if self.busy else 'normal')
        for combo in self.combos: combo.configure(state='disabled' if self.busy else 'readonly')
        self.recognise_button.configure(state='normal' if self.image is not None and not self.busy else 'disabled')
        self.export_button.configure(state='normal' if self.result is not None and not self.busy else 'disabled')

    def clear_result(self):
        self.result=None; self.settings=None; self.prediction.set('No prediction yet')
        self.details.set('Select Recognise to process this image.'); self.digits_canvas.delete('all')
        self.redraw(); self.refresh_controls()

    def settings_changed(self,event=None):
        self.clear_result(); self.status.set('Settings changed. Run recognition again.')

    def set_image(self,image,origin):
        self.image=image; self.origin=origin
        self.source_text.set('Generated image | Requested: '+origin['requested_text'] if origin['route']=='generated' else 'Photo | '+Path(origin['file']).name)
        self.clear_result()

    def start_job(self,kind,function):
        if self.busy:return
        self.busy=True;self.status.set('Generating image...' if kind=='generate' else ('Loading image...' if kind=='load' else 'Recognising handwriting...'))
        self.refresh_controls()
        def worker():
            try:self.messages.put((kind,function(),None))
            except Exception as error:self.messages.put((kind,None,str(error)))
        self.job_thread=threading.Thread(target=worker,daemon=True); self.job_thread.start()

    def load_dialog(self):
        path=filedialog.askopenfilename(parent=self.root,title='Choose a handwriting photo',initialdir=ROOT/'data',filetypes=[('Image files','*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),('All files','*.*')])
        if not path:return
        def load():
            return open_photo(path),{'route':'file','file':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        self.clear_result();self.start_job('load',load)

    def folder_dialog(self):
        path=filedialog.askdirectory(parent=self.root,title='Folder of labelled digit images',initialdir=ROOT/'data')
        if path:self.folder.set(path)

    def generate(self):
        folder=self.folder.get();text=self.number.get().strip()
        self.clear_result();self.start_job('generate',lambda:generate_number(folder,text))

    def recognise(self):
        if self.image is None or self.busy:return
        image=self.image.copy(); settings={'model':self.model.get(),'method':self.method.get(),'representation':self.representation.get()}
        self.clear_result()
        def work():
            from hnrs_ml.predict import DigitPredictor
            from segment_digits import recognise_number
            model=settings['model']
            if model not in self.predictors:self.predictors[model]=DigitPredictor(ROOT/'experiments/baseline'/f'{model}.pt')
            checkpoint=ROOT/'experiments/baseline'/f'{model}.pt'
            settings['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result=recognise_number(image,self.predictors[model],settings['method'],settings['representation'])
            return result,settings
        self.start_job('recognise',work)

    def poll(self):
        if self.closed:return
        try:kind,value,error=self.messages.get_nowait()
        except queue.Empty:pass
        else:
            self.busy=False
            if error:
                self.status.set('Could not complete: '+error)
                messagebox.showerror('Could not complete',error,parent=self.root)
            elif kind in ('generate','load'):
                self.set_image(*value)
                warnings=value[1].get('warnings',[])
                self.status.set('Image ready. Select Recognise.'+(' Source warning: '+'; '.join(warnings) if warnings else ''))
            else:
                self.result,self.settings=value; prediction=self.result['prediction'];seg=self.result['segmentation']
                self.prediction.set(prediction['text'] if prediction['text'] else 'No digits detected')
                self.details.set(f"{len(seg.boxes)} digit(s) | {self.settings['model'].upper()} | {self.settings['method']} | {self.settings['representation']}")
                self.status.set('; '.join(seg.warnings) if seg.warnings else 'Recognition finished. Inspect the crops and compare the result with the handwriting.')
                self.redraw();self.draw_digits()
            self.refresh_controls()
        self.poll_id=self.root.after(80,self.poll)

    def redraw(self):
        if not hasattr(self,'output_canvas'):return
        self.photos=[]
        detected=annotated_image(self.result['segmentation']) if self.result else None
        for canvas,image,placeholder in [(self.input_canvas,self.image,'Open or generate an image'),(self.output_canvas,detected,'Digit regions appear after recognition')]:
            canvas.delete('all');w=max(50,canvas.winfo_width());h=max(50,canvas.winfo_height())
            if image is None:canvas.create_text(w/2,h/2,text=placeholder,fill='#8493a8',font=('Segoe UI',10),width=w-40)
            else:
                image=image.copy();image.thumbnail((max(1,w-20),max(1,h-20)))
                photo=tk_image(image,self.root);self.photos.append(photo)
                canvas.create_image(w/2,h/2,image=photo)

    def draw_digits(self):
        self.digit_photos=[];self.digits_canvas.delete('all')
        if not self.result:return
        for index,(array,item) in enumerate(zip(self.result['prepared'],self.result['prediction']['digits'])):
            image=Image.fromarray(np.rint(array*255).astype('uint8')).resize((64,64),Image.Resampling.NEAREST)
            photo=tk_image(image,self.root);self.digit_photos.append(photo)
            x=index*128+8
            self.digits_canvas.create_rectangle(x,0,x+118,106,fill='#f3f6fb',outline='')
            self.digits_canvas.create_image(x+27,8,image=photo,anchor='nw')
            self.digits_canvas.create_text(x+59,87,anchor='center',text=f"{item['digit']}   ·   {item['confidence']:.1%}",fill='#334155',font=('Segoe UI',10,'bold'))
        self.digits_canvas.configure(scrollregion=(0,0,max(1,len(self.digit_photos))*128,110))

    def export_dialog(self):
        if not self.result or self.busy:return
        folder=filedialog.askdirectory(parent=self.root,title='Save a new evidence folder here',initialdir=ROOT/'evidence')
        if not folder:return
        try:
            destination=export_result(folder,self.image,self.origin,self.result,self.settings)
            self.status.set('Saved evidence: '+str(destination))
        except (OSError,ValueError) as error:messagebox.showerror('Export failed',str(error),parent=self.root)

    def close(self):
        self.closed=True
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    root=tk.Tk(); HNRSApp(root);root.mainloop()


if __name__=='__main__':main()
