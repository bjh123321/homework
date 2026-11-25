"""Interactive Python instrument for inspecting stored runs and individual CSV files."""
import argparse
from pathlib import Path
import sqlite3
import subprocess
import sys

import matplotlib
import numpy as np
from matplotlib.widgets import Button, TextBox

from bearing_monitor import FS, N, T, simulated, waveform, diagnostic, alarms

HERE=Path(__file__).resolve().parent
matplotlib.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                            "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,
                            "axes.labelsize":9,"axes.titlesize":11,"legend.fontsize":8})

class Instrument:
    def __init__(self,out,minute=480):
        import matplotlib.pyplot as plt
        self.out=Path(out)
        if not (self.out/"monitor.sqlite").is_file():
            raise ValueError("请先运行批处理生成 monitor.sqlite")
        self.db=sqlite3.connect(str(self.out/"monitor.sqlite"))
        self.source=self.db.execute("SELECT source FROM metadata").fetchone()[0]
        self.names=[r[0] for r in self.db.execute("SELECT DISTINCT bearing FROM features ORDER BY bearing")]
        self.idx=0
        self.minute=minute
        self.custom=None
        self.fig,self.axes=plt.subplots(2,3,figsize=(13,7.6))
        self.fig.canvas.mpl_connect("close_event",lambda _:self.db.close())
        self.fig.subplots_adjust(left=.065,right=.978,top=.79,bottom=.21,hspace=.60,wspace=.35)
        self.fig.patch.set_facecolor("#f4f6f8")
        self.heading=self.fig.text(.057,.937,"",fontsize=20,weight="bold",color="#17344a")
        self.info=self.fig.text(.058,.876,"",fontsize=10,color="#324f60")
        self.status=self.fig.text(.058,.035,"",fontsize=9,color="#385863")
        self.widgets=[]
        for xpos,width,label,callback in [
            (.06,.107,"上一轴承",lambda _:self.move(-1)),(.18,.107,"下一轴承",lambda _:self.move(1)),
            (.51,.12,"打开双通道 CSV",self.open_csv),(.64,.13,"导出当前图 PNG",self.save),
            (.78,.14,"批量导入实测目录",self.batch)]:
            b=Button(self.fig.add_axes([xpos,.075,width,.044]),label,color="#e3edf1",hovercolor="#c9e3eb")
            b.on_clicked(callback)
            self.widgets.append(b)
        self.number=TextBox(self.fig.add_axes([.385,.075,.085,.044]),"记录号  ",initial=str(minute))
        self.number.on_submit(self.choose)
        self.render()

    def trend(self):
        return self.db.execute("SELECT minute,rms_h,rms_v,signature FROM features WHERE bearing=? ORDER BY minute",
                               (self.names[self.idx],)).fetchall()
    def move(self,step):
        self.idx=(self.idx+step)%len(self.names)
        self.custom=None
        self.render()
    def choose(self,text):
        try:
            self.minute=int(text)
            self.custom=None
            self.render()
        except (ValueError,OSError) as exc:
            self.notice(str(exc))
    def notice(self,msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
    def open_csv(self,event):
        from tkinter import filedialog
        path=filedialog.askopenfilename(filetypes=[("双通道 CSV","*.csv")])
        if path:
            try:
                self.custom=(Path(path).name,waveform(path))
                self.render()
            except (ValueError,OSError) as exc:
                self.notice(str(exc))
    def batch(self,event):
        from tkinter import filedialog
        root=filedialog.askdirectory(title="选择 XJTU-SY 原始数据目录")
        if not root:
            return
        out=filedialog.askdirectory(title="选择独立的实测输出目录")
        if out:
            try:
                from bearing_monitor import collect
                self.notice("正在分批导入数据…")
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                collect(out,root)
                subprocess.Popen([sys.executable,str(HERE/"bearing_view.py"),"--out",out])
                self.notice("批处理完成，已打开实测结果窗口")
            except Exception as exc:
                self.notice(str(exc))
    def save(self,event):
        from tkinter import filedialog
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png")])
        if path:
            self.fig.savefig(path,dpi=160)
            self.notice("图像已导出")
    def render(self):
        rows=self.trend()
        mins=np.array([r[0] for r in rows])
        rms=np.array([r[1:3] for r in rows])
        i=int(np.argmin(abs(mins-self.minute)))
        minute=int(mins[i])
        self.number.eventson=False
        self.number.set_val(str(minute) if not self.custom else "—")
        self.number.eventson=True
        name=self.names[self.idx]
        sim=self.source=="sim-v1"
        if self.custom:
            label,x=self.custom
        elif sim:
            x=simulated(int(name[1])-1,name.split("_")[1],minute)
            label=f"仿真  ·  {name}  ·  第 {minute} 条记录"
        else:
            x=waveform(Path(self.source)/rows[i][3].split("/",2)[2])
            label=f"实测目录  ·  {name}  ·  第 {minute} 条记录"
        f,fft,fw,psd,env,ft,ts,z=diagnostic(x)
        trend,time,amp,power,envelope,tf=self.axes.flat
        for ax in self.axes.flat:
            ax.clear()
            ax.grid(alpha=.16)
        lim,_,first=alarms(mins,rms)
        if self.custom:
            trend.set_axis_off()
            trend.text(.1,.5,"独立 CSV\n无历史记录",transform=trend.transAxes,fontsize=14)
        else:
            trend.set_axis_on()
            trend.plot(mins,rms)
            for v,color in zip(lim,("C0","C1")):
                trend.axhline(v,ls=":",color=color,lw=.9)
            if "OR" in first:
                trend.axvline(first["OR"],color="#97532c",ls="--",lw=1)
            trend.axvline(minute,color="#333",lw=.8)
            trend.set(xlabel="记录号（每分钟一条）",ylabel="去均值 RMS / g")
        trend.set_title("寿命趋势与预警阈值",loc="left")
        time.plot(T[:1024]*1000,x[:1024],lw=.8)
        time.legend(["水平","垂直"],loc="upper right",ncol=2)
        time.set(xlabel="时间 / ms",ylabel="加速度 / g",title="双通道波形  ·  前 40 ms")
        amp.plot(f,fft,lw=.65)
        amp.set(xlim=(0,6000),xlabel="频率 / Hz",ylabel="峰值幅度 / g",title="单边幅值谱")
        power.semilogy(fw,np.maximum(psd,1e-15),lw=.75)
        power.set(xlabel="频率 / Hz",ylabel="PSD / (g²/Hz)",title="Welch 功率谱密度")
        mask=(f>0)&(f<=600)
        envelope.plot(f[mask],env[mask],lw=.8)
        envelope.set(xlabel="调制频率 / Hz",ylabel="包络谱幅度 / g",title="2–5 kHz 带通解调")
        tf.pcolormesh(ts,ft,20*np.log10(np.maximum(z,1e-12)),shading="auto",cmap="viridis",vmin=-70,vmax=-15)
        tf.set(xlabel="时间 / s",ylabel="频率 / Hz",title="水平通道 STFT  ·  −70 至 −15 dB")
        self.heading.set_text("滚动轴承双通道振动监测")
        self.info.set_text(f"{label}     |     采样率 25.6 kHz     |     记录长 1.28 s")
        self.notice("单文件诊断" if self.custom else f"水平 / 垂直阈值 {lim[0]:.4f} / {lim[1]:.4f} g     ｜     任一通道预警 {first.get('OR','无')}     ｜     双通道确认 {first.get('AND','无')}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=HERE/"bearing_results")
    p.add_argument("--snapshot",type=Path)
    args=p.parse_args()
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    app=Instrument(args.out)
    if args.snapshot:
        app.fig.savefig(args.snapshot,dpi=145)
    else:
        plt.show()

