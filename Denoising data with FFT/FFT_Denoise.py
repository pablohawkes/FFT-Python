# Denoising data with Fast Fourier Transform
# Original from Steve Brunton:
# https://www.youtube.com/watch?v=s2K1JfNR7Sc
# http://databookuw.com/databook.pdf 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

#####################################################
# Create simple signal with several frequencies and add noise:
    
dt = 0.0001 # sample frequence: 1000Hz >> max signal frequence <= 500 Hz
t = np.arange(0,1, dt) 
f = 1 * np.cos(1 * np.pi * 10 * t ) + 2 * np.sin(2 * np.pi * 20 * t ) + \
    1.5 * np.sin(2 * np.pi * 40 * t )  + 5 * np.sin(2 * np.pi * 50 * t ) + \
    2.1 * np.cos (2 * np.pi * 80 * t ) + 2.3 * np.cos(2 * np.pi * 120 * t ) + \
    5 * np.sin(2 * np.pi * 150 * t ) + 5 * np.sin(2 * np.pi * 180 * t )

f_clean = f
f = f + 5 * np.random.randn(len(t))


#####################################################
#create frequency array and calculate FFT:
    
n = len(t)
fhat = np.fft.fft(f,n)
PSD = (fhat * np.conj(fhat)) / n #Power spectrum
freq = (1 / (dt * n)) * np.arange (n) # create X-axis of frequences
L = np.arange(1, np.floor(n/2), dtype='int') #Only plot the first half of freqs

#####################################################
#calculate PSD threshold (sort of):
#this section tries to calculate PSD mean using all but outliers values. 

#calculate mean and standard deviation of PSD array (std dev).    
arrayToCalculate = PSD.real
mean = np.mean(PSD.real)
standard_deviation = np.std(arrayToCalculate)
distance_from_mean = abs(arrayToCalculate - mean)

max_deviations = 4 #1.96: 95% confidence / 3: 99.7% confidence

not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = arrayToCalculate[not_outlier]
avgPSD = np.mean(no_outliers)
stdDevPSD = np.std(no_outliers)

#avgPSD = np.mean(PSD.real)
#stdDevPSD = np.std(PSD.real)
print("Mean of PSD without outliers: " + str(avgPSD) )
print("Standar deviation of PSD without outliers: " + str(stdDevPSD) )


# calculate threshold using mean and std dev:
thresMax = avgPSD + max_deviations * stdDevPSD
#thresMin = avgPSD - max_deviations * stdDevPSD

print("PSD Upper threshold (mean + " + str(max_deviations) + " std dev): " + str(thresMax))

#####################################################
#Filter PSD array using threshold:
#Threshold shoud only consider BIG outliers (dominant frequences).

indices = PSD > thresMax
PSDClean = PSD * indices
fhat = indices * fhat
ffilt = np.fft.ifft (fhat)

#####################################################
# Plots:

plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size': 18})

fig,axs = plt.subplots(3,1)

plt.sca(axs[0])
plt.plot(t, f, color = 'c', LineWidth = 1.5, label='Noisy')
plt.plot(t, f_clean, color = 'k', LineWidth = 3, label='Clean')
#plt.xlim(0, 0.3)#plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color = 'c', LineWidth = 1.5, label='PSD/Frequence')
plt.axhline(thresMax, color='g', linestyle='--', LineWidth = 0.5, label='PSD threshold max')
#plt.axhline(thresMin, color='r', linestyle='--', LineWidth = 0.5, label='PSD threshold min')
plt.xlim(0,250)#plt.xlim(L[0],L[-1])
#plt.ylim(-300,400)
plt.yscale('log')
plt.legend()

plt.sca(axs[2])
plt.plot(t, ffilt, color = 'g', LineWidth = 1.5, label='Filtered')
#plt.xlim(0, 0.3)#plt.xlim(t[0], t[-1])
plt.legend()

plt.show()

#####################################################
#Converto to sound:
    
scaledC = np.int16(f_clean /np.max(np.abs(f_clean)) * 32767)
write('C:\\temp\\f_clean.wav', 44100, scaledC)
scaledF = np.int16(ffilt/np.max(np.abs(ffilt)) * 32767)
write('C:\\temp\\f_filt.wav', 44100, scaledF)
scaledD = np.int16(f/np.max(np.abs(f)) * 32767)
write('C:\\temp\\f_dirty.wav', 44100, scaledD)

#write('C:\\temp\\f_clean.wav', 44100, f_clean)
#write('C:\\temp\\f_filt.wav', 44100, ffilt)
#write('C:\\temp\\f_dirty.wav', 44100, f)

#####################################################