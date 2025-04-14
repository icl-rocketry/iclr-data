import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from scipy.misc import derivative
from scipy.interpolate import CubicSpline

#----------------------------------------------------------------
#--------------------- Parameter Definition ---------------------
#----------------------------------------------------------------

T_nominal = 2400 #Nominal high thrust [N]
T_startup = 500 #Startup thrust
Percent_tolerance = 5 
T_tolerance = T_nominal*(Percent_tolerance/100)
Percent_low_throttle = 40
T_low = (T_nominal - T_tolerance) * (Percent_low_throttle/100) - T_tolerance

g = 9.81

Isp = 190 #Specific impulse in seconds

N2O_mass = 7.5 #Maximum nitrous mass [kg]

T_gradient_1 = 3000 #Gradient of 1st throttle-up rate of change of thrust [N/s]
T_gradient = 1500 #Gradient of rate of change of thrust [N/s]
#Combinations: T_grad = 1300, t_low = 2.5; T_grad = 1050, t_low = 2

t_duration = 12
dt = 0.01

t = np.arange(0, t_duration + dt, dt)

t_throttle_up_1 = 0.5
t_high_thrust = 4.4
t_low_thrust = 3

t_throttle_down = t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1 + t_high_thrust

t_throttle_up_2 = t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust

T = np.zeros(len(t))

#----------------------------------------------------------------
#-------------------- Create Thrust Profile ---------------------
#----------------------------------------------------------------
for i in range(0,len(t)):
    if t[i] < t_throttle_up_1: #Startup
        T[i] = T_startup
    elif t[i] < (t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1) : #Throttle Up 1
        T[i] = T_startup + T_gradient_1*(t[i] - t_throttle_up_1)
    elif t[i] < (t_throttle_down): #High Thrust
        T[i] = T_nominal
    elif t[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient): #Throttle Down
        T[i] = T_nominal - (t[i] - t_throttle_down) * T_gradient
    elif t[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust): #Low Thrust
        T[i] = T_low
    elif t[i] < (t_throttle_up_2 + (T_nominal-T_low)/T_gradient): #Throttle Up 2
        T[i] = T_low + T_gradient*(t[i] - t_throttle_up_2)
    else:
        T[i] = T_nominal

T_max = T + T_tolerance
T_min = T - T_tolerance

data = pd.read_csv('20240210_THANOS-A_HOT-FIRE_1_A_RAW-DATA-BACKEND.csv', index_col=1)
#data2 = pd.read_csv('20230917-THANOS-HOT-FIRE-4-UPPER-FEED.csv', index_col=1)
#data3 = pd.read_csv('20230917-THANOS-HOT-FIRE-4-GSS.csv', index_col=1)

testname = '20240210_THANOS-A_HOT-FIRE_1_A'

t_start = 142
t_ignition = 144.2
t_end = 160

#print(data)

#data.columns = ['P_N2_Tank','Ch2','P_Chamber','P_Fuel_Tank','Ch4','Ch5','Rocket_Mass','Thrust','P_Fuel_Inlet','P_Ox_Tank','P_Fuel_Injector','Ch10','Ch11','Fuel_Flow_Rate','Ch13','Time']

data.rename(columns={'ch0sens': 'P_N2_Tank',
                     'ch2sens': 'P_Chamber',
                     'ch3sens': 'P_Fuel_Tank',
                     'ch6sens': 'Rocket_Mass',
                     'ch7sens': 'Thrust',
                     'ch8sens': 'P_Fuel_Inlet',
                     'ch9sens': 'P_Ox_Tank',
                     'ch10sens': 'P_Fuel_Injector',
                     'temp0': 'TC1',
                     'temp1': 'TC2',
                     'temp2': 'Fuel_Flow_Rate',
                     'temp3': 'TC3',
                     'system_time': 'Time'}, inplace=True)


data.loc[:,'Time'] = data.loc[:,'Time']/1000
Time = data.loc[:,'Time']
data.loc[:,'Time'] = data.loc[:,'Time'] - Time.iloc[0]


data = data.loc[data['Time'] > t_start]
data = data.loc[data['Time'] < t_end]
data.loc[:,'Time'] = data.loc[:,'Time'] - t_ignition


Smoothed_thrust = -gaussian_filter(data.loc[:,'Thrust'],sigma=4)
data.loc[:,'Thrust'] = -data.loc[:,'Thrust']

Impulse = integrate.simpson(data.loc[:,'Thrust'], data.loc[:,'Time'])
print('Total Impulse: ', Impulse, ' Ns')

plt.plot(data.loc[:,'Time'],Smoothed_thrust, label = 'Smoothed_Thrust',color='black')
plt.scatter(data.loc[:,'Time'],data.loc[:,'Thrust'], label = 'Raw Thrust', marker = 'o', color='red', s=1)
plt.plot(t,T, label = 'Nominal_Thrust_Trace',color='blue')
plt.plot(t,T_max, label = 'Max_Thrust_Trace',color='orange')
plt.plot(t,T_min, label = 'Min_Thrust_Trace',color='green')

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Thrust [N]')
plt.xlim(1.5,11.6)
plt.legend()
plt.title(testname +'_Thrust-Curve')
plt.savefig(testname +'_Thrust-Curve-Comparison.png', dpi=300)
plt.show()

Ox_mass = (-data.loc[:,'Rocket_Mass']-120)/9.81 
Smoothed_ox_mass = gaussian_filter(Ox_mass,sigma=5)
plt.plot(data.loc[:,'Time'],Smoothed_ox_mass, label = 'Smoothed Ox Mass')
plt.scatter(data.loc[:,'Time'],Ox_mass, label = 'Ox Mass', marker = 'o', color='red', s=1)

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.legend()
plt.title(testname +'_Ox-Tank-Mass')
plt.savefig(testname +'_Ox-Tank-Mass.png', dpi=300)
plt.show()


plt.plot(data.loc[:,'Time'],data.loc[:,'P_N2_Tank'], label = 'N2 Tank')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Ox_Tank'], label = 'Ox Tank')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Tank'], label = 'Fuel Tank')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Inlet'], label = 'Ox Injector')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Chamber'], label = 'Chamber')

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.legend()
plt.title(testname +'_High-Pressures')
plt.savefig(testname +'_High-Pressures.png', dpi=300)
plt.show()

plt.plot(data.loc[:,'Time'],data.loc[:,'P_Ox_Tank'], label = 'Ox Tank')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Tank'], label = 'Fuel Tank')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Inlet'], label = 'Ox Injector')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
plt.plot(data.loc[:,'Time'],data.loc[:,'P_Chamber'], label = 'Chamber')

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.legend()
plt.title(testname +'_Low-Pressures')
plt.savefig(testname +'_Low-Pressures.png', dpi=300)
plt.show()


Smoothed_Flow_Rate = gaussian_filter(data.loc[:,'Fuel_Flow_Rate'], sigma=2)

fig, ax1 = plt.subplots()


ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Pressure [bar]')
ax1.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Tank'], label = 'Fuel Tank')
ax1.plot(data.loc[:,'Time'],data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
ax1.plot(data.loc[:,'Time'],data.loc[:,'P_Chamber'], label = 'Chamber')
ax1.tick_params(axis='y')
ax1.legend()


plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'black'
ax2.set_ylabel('Volumetric Flow Rate [L/s]', color=color)  # we already handled the x-label with ax1
ax2.plot(data.loc[:,'Time'],Smoothed_Flow_Rate, label = 'Smoothed Fuel Flow Rate', color='black')
ax2.scatter(data.loc[:,'Time'],data.loc[:,'Fuel_Flow_Rate'], label = 'Fuel Flow Rate', marker = 'o', color='red', s=1)
ax2.set_ylim(0,1)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend()

fig.tight_layout()# otherwise the right y-label is slightly clipped
plt.title(testname + '_Flow-Rate-vs-Pressure')
plt.savefig(testname +'_Flow-Rate-vs-Pressure.png', dpi=300)
plt.show()

cs = CubicSpline(data.loc[:,'Time'],Smoothed_ox_mass,bc_type='clamped')
m_dot_ox = -derivative(cs, data.loc[:,'Time'])
rhoFuel = 786 #fuel density in kg/m^3
m_dot_fuel = (Smoothed_Flow_Rate/1000)*rhoFuel
plt.plot(data.loc[:,'Time'],m_dot_ox, label = 'm_dot_ox')
plt.plot(data.loc[:,'Time'],m_dot_fuel, label = 'm_dot_fuel')

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Mass Flow Rate [kg/s]')
plt.xlim(0,15)
plt.ylim(0,1)
plt.legend()
plt.title(testname +'_Prop-Mass-Flow')
plt.savefig(testname +'_Prop-Mass-Flow.png', dpi=300)
plt.show()

OF = m_dot_ox/m_dot_fuel
Isp = Smoothed_thrust/(g*(m_dot_ox+m_dot_fuel))

fig, ax1 = plt.subplots()

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Estimated OF Ratio')
ax1.plot(data.loc[:,'Time'],OF,color='black',label='OF Ratio')
ax1.set_xlim(0,15)
ax1.set_ylim(0,3)
ax1.tick_params(axis='y')

plt.legend()

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Estimated Isp [s]', color=color)  # we already handled the x-label with ax1
ax2.plot(data.loc[:,'Time'],Isp, color='red',label='Isp')
ax2.set_ylim(0,250)
ax2.tick_params(axis='y', labelcolor=color)
plt.legend()

plt.title(testname +'_Engine-OF-&-Isp')
plt.savefig(testname +'_Engine-OF-&-Isp', dpi=300)
plt.legend()
plt.show()