import tkinter as tk
import tkinter.ttk as ttk
import datetime as dt

class TimeVar(tk.Variable):
    def __init__(self, *args, **kwargs):
        tk.Variable.__init__(self, *args, **kwargs)
        self.set(dt.time())
    def get(self):
        timeNumber = tk.Variable.get(self)
        hours = int(timeNumber // 3600)
        minutes = int((timeNumber - 3600 * hours) // 60)
        seconds = int((timeNumber - 3600 * hours - 60 * minutes) // 1)
        microsecond = int((timeNumber - 3600 * hours - 60 * minutes - seconds) * 1000000)
        return dt.time(hour=hours, minute=minutes, second=seconds, microsecond=microsecond)
    def set(self, time):
        # Expects a datetime.time object
        timeNumber = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1000000
        tk.Variable.set(self, timeNumber)

class TimeEntry(ttk.LabelFrame):
    AM = "AM"
    PM = "PM"
    AMPMs = [AM, PM]
    def __init__(self, parent, *args, style=None, timevar=None, **kwargs):
        self.parent = parent
        self.timevar = timevar
        if style is None:
            self.style = ttk.Style()
            self.style.theme_use('default')
        else:
            self.style = style
        self.style.configure('Sub.TEntry', borderwidth=0, background='white')
        self.style.configure('Sub.TLabel', borderwidth=0, background='white')
        self.style.configure('Sub.TCombobox', borderwidth=0, background='white')
        self.style.configure('Outer.TFrame', borderwidth=3, bordercolor='black')
        ttk.LabelFrame.__init__(self, self.parent, *args, style='Outer.TFrame', **kwargs)
        self.hourVar = tk.StringVar()
        validateHourID = self.parent.register(TimeEntry.validateHour)
        validateMinuteID = self.parent.register(TimeEntry.validateMinute)
        self.hour = ttk.Entry(self, textvariable=self.hourVar, width=2, style='Sub.TEntry', validate='key', validatecommand=(validateHourID, '%P'))
        self.colon = ttk.Label(self, text=":", style='Sub.TLabel')
        self.minuteVar = tk.StringVar()
        self.minute = ttk.Entry(self, textvariable=self.minuteVar, width=2, style='Sub.TEntry', validate='key', validatecommand=(validateMinuteID, '%P'))
        self.AMPMVar = tk.StringVar()
        self.AMPMVar.set(TimeEntry.AM)
        self.AMPM = ttk.Combobox(self, values=TimeEntry.AMPMs, textvariable=self.AMPMVar, width=3, style='Sub.TCombobox')

        self.hour.grid(row=0, column=0)
        self.colon.grid(row=0, column=1)
        self.minute.grid(row=0, column=2)
        self.AMPM.grid(row=0, column=3)

        self.minuteVar.trace('w', self.change)
        self.hourVar.trace('w', self.change)
        self.AMPMVar.trace('w', self.change)

        if self.timevar is not None:
            self.timevar.trace('w', self.varChange)

        self.allowCallback = True

    def change(self, *args):
        # User changed the time by typing/clicking
        if self.allowCallback and self.timevar is not None:
            self.allowCallback = False
            try:
                self.timevar.set(self.get())
            except:
                print('uh oh!')
            finally:
                self.allowCallback = True

    def varChange(self, *args):
        # The time is being changed via the attached TimeVar - update the entries to reflect that
        if self.allowCallback and self.timevar is not None:
            self.allowCallback = False
            try:
                self.set(self.timevar.get())
            except:
                print("uh oh")
                pass
            finally:
                self.allowCallback = True

    def validateHour(hour):
        if len(hour) == 0:
            return True
        try:
            int(hour)
        except ValueError:
            return False

        if len(hour) > 2:
            return False
        if int(hour) < 1 or int(hour) > 12:
            return False
        return True

    def validateMinute(minute):
        if len(minute) == 0:
            return True
        try:
            int(minute)
        except ValueError:
            return False

        if len(minute) > 2:
            return False
        if int(minute) < 0 or int(minute) > 59:
            return False
        return True

    def get(self):
        return dt.time(hour=self.getHour(military=True), minute=self.getMinute())

    def getHour(self, military=False):
        hour = self.hourVar.get()
        if len(hour) == 0:
            hour = 0
        else:
            hour = int(hour) + 12*self.getPM()*military
        return hour

    def getMinute(self):
        minute = self.minuteVar.get()
        if len(minute) == 0:
            minute = 0
        else:
            minute = int(minute)
        return minute

    def getAMPM(self):
        return self.AMPMVar.get()

    def getAM(self):
        return self.AMPMVar.get() == TimeEntry.AM

    def getPM(self):
        return self.AMPMVar.get() == TimeEntry.PM

    def set(self, time):
        # time = datetime.time object
        self.hourVar.set(((time.hour - 1) % 12) + 1)
        self.minuteVar.set(time.minute)
        self.AMPMVar.set(TimeEntry.AMPMs[time.hour // 12])

    def now(self):
        self.set(dt.datetime.now())
