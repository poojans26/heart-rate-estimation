import cv2
import numpy as np
import copy
import time
from sklearn.decomposition import FastICA
from scipy.fftpack import fft
import matplotlib.pyplot as plt

#detect faces using Haar cascade classifier for a video frame
def detectFacesInImage(videoframe):
    # convert the image to grayscale
    gray = cv2.cvtColor(videoframe, cv2.COLOR_BGR2GRAY)

    rectFrame = np.zeros(videoframe.shape, videoframe.dtype)
    # facesFound = np.empty([1,4])
    # create a cascade classifier to detect frontal faces and profile faces using haar cascade classifier
    faceCascadeClassifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    profileCascadeClassifier = cv2.CascadeClassifier('haarcascade_profileface.xml')

    #create a cascade classifier to detect eye on the face ROI, this is for reducing the false detection of faces
    eyeCascadeClassifier = cv2.CascadeClassifier('haarcascade_eye.xml')

    # detect faces
    faces = faceCascadeClassifier.detectMultiScale(gray, 1.3, 5, 0, (70, 70))
    facesFound = np.array(faces,dtype=int)

    if(len(facesFound) == 0):
        faces = profileCascadeClassifier.detectMultiScale(gray, 1.3, 5, 0, (70, 70))
        facesFound = np.array(faces,dtype=int)

    totalFaces = []
    faceRectangles = []
    if(len(facesFound) != 0) :
        # draw the faces
        for face in facesFound:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            eyeROIGray = gray[y:y + h, x:x + w]
            eyeROIColor = rectFrame[y:y + h, x:x + w]
            eyes = eyeCascadeClassifier.detectMultiScale(eyeROIGray, 1.3, 5, 0)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(eyeROIColor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            isEyesDetected = len(eyes) > 0

            #confirm it is a face and post process for heart rate
            if(isEyesDetected) :
                cv2.rectangle(rectFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                faceROI = copy.copy(videoframe[y:y + h, x:x + w])
                totalFaces.append(faceROI)
                faceRectangles.append([x,y,w,h])

    videoframe = cv2.add(videoframe,rectFrame)
    return videoframe,totalFaces


#collect data for estimation
def collectDataForHeartRateEstimate(traces, detectedFace):

    #crop face ROI
    frows = detectedFace.shape[0]
    fcols = detectedFace.shape[1]

    # croppedROI = copy.copy(detectedFace[0:frows - 10,20:fcols - 20])
    croppedROI = copy.copy(detectedFace[0:30,20:fcols - 20])


    videoWindow2 = "Cropped face"
    cv2.namedWindow(videoWindow2, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(videoWindow2, croppedROI)

    #get the normalized RGB traces and the time elapsed
    red, green, blue = getRGBTraces(croppedROI)
    elapsed_time = time.time() - start_time

    #add in a list entry
    singleReading = []
    singleReading.append(elapsed_time)
    singleReading.append(red)
    singleReading.append(green)
    singleReading.append(blue)

    traces = addReadingToTraces(traces, singleReading)
    print singleReading

    return traces

# add data to the traces buffer
def addReadingToTraces(traces,singleReading):
    # add the reading to the existing list of traces
    #maintain the length of the list as the threshold given
    if len(traces) < thresholdTraces:
        traces.append(singleReading)
    else:
        traces.pop(0)
        traces.append(singleReading)

    return traces

#get the normalized RGB traces
def getRGBTraces(faceROI):

    #separate the pixel values and average them to get the trace values
    b, g, r = cv2.split(faceROI)

    sum_blue = 0.0
    sum_green = 0.0
    sum_red = 0.0

    frows = faceROI.shape[0]
    fcols = faceROI.shape[1]
    total_pixels = frows * fcols

    #find mean of the ROI
    for row in range(0,frows):
        for col in range(0,fcols):
            blue_value = b.item(row,col)
            green_value = g.item(row,col)
            red_value = r.item(row,col)

            sum_blue += blue_value
            sum_green += green_value
            sum_red += red_value

    mean_blue = sum_blue / total_pixels
    mean_green = sum_green / total_pixels
    mean_red = sum_red / total_pixels

    sq_blue = 0.0
    sq_green = 0.0
    sq_red = 0.0

    #find standard deviation
    for row in range(0,frows):
        for col in range(0,fcols):
            blue_value = b.item(row, col)
            green_value = g.item(row, col)
            red_value = r.item(row, col)

            sq_blue += np.square(blue_value - mean_blue)
            sq_green += np.square(green_value - mean_green)
            sq_red += np.square(red_value - mean_red)

    std_blue = np.sqrt(sq_blue / total_pixels)
    std_green = np.sqrt(sq_green / total_pixels)
    std_red = np.sqrt(sq_red / total_pixels)

    total_blue_val = 0.0
    total_green_val = 0.0
    total_red_val = 0.0

    #find norm RGB trace
    for row in range(0,frows):
        for col in range(0,fcols):
            norm_blue_value = (b.item(row, col) - mean_blue) / std_blue
            norm_green_value = (g.item(row, col) - mean_green) / std_green
            norm_red_value = (r.item(row, col) - mean_red) / std_red

            total_blue_val += norm_blue_value
            total_green_val += norm_green_value
            total_red_val += norm_red_value

    norm_blue_trace = total_blue_val / total_pixels
    norm_green_trace = total_green_val / total_pixels
    norm_red_trace = total_red_val / total_pixels

    # return norm_red_trace, norm_blue_trace, norm_green_trace
    return mean_red,mean_green,mean_blue

# find the bpm for one component from ICA
def getHeartRateForComponent(reconstructedComponent, times, tracelength, frameRate):

    #get equally spaced intervals from the elapsed time
    # intervals = np.linspace(times[0], times[-1], tracelength)
    #
    # # fit a curve over the component values and interpolate to find the value for the intervals above
    # interpolatedValues = np.interp(intervals, times, reconstructedComponent)
    #
    # #smoothen and standardize the interpolated values
    # interpolatedValues = np.hamming(tracelength) * interpolatedValues
    # interpolatedValues = interpolatedValues - np.mean(interpolatedValues)

    #run FFT on the interpolated values.rfft gives only the positive values from the interpolated values since the FFT is a symmetric curve
    # fftValues = np.fft.rfft(interpolatedValues)
    fftValues = np.fft.rfft(reconstructedComponent)

    #get the power of the FFT
    powerFFT = np.abs(fftValues)
    #descending sorted index
    sortedidx = powerFFT.argsort()[::-1][:len(powerFFT)]

    #construct frequencies at equally spaced intervals. the length is divided by half as we get only half the values .
    # f = 1/t
    # multiply by 60 as the frequency to bpm mapping indicates [0.75,4] Hz maps to [45,240] bpm
    freqsComponent = (frameRate / tracelength) * np.arange(tracelength / 2 + 1)
    freqsComponent = 60 * freqsComponent

    hr = 0.0
    for idx in sortedidx:
        if(freqsComponent[idx] > 50 and freqsComponent[idx] < 150 ):
            hr = freqsComponent[idx]
            break

    #filter by human heart bpm range both the frequency and power
    # index = np.where((freqsComponent > 50) & (freqsComponent < 150))
    # filteredFreq = freqsComponent[index]
    # filteredPower = powerFFT[index]
    #
    # #the bpm corresponds to the frequency for the maximum power
    # maxPower = np.argmax(filteredPower)
    # hr = filteredFreq[maxPower]


    return hr

#method called to get the heart rate values after collecting the data
def estimateHeartRate(traces,frameRate):
    traceArray = np.array(traces)

    redTrace = traceArray[:,1]
    redmean = np.mean(redTrace)
    redstd = np.std(redTrace)
    norm_red = (redTrace - redmean ) / redstd

    greenTrace = traceArray[:,2]
    greenmean = np.mean(greenTrace)
    greenstd = np.std(greenTrace)
    norm_green = (greenTrace - greenmean) / greenstd

    blueTrace = traceArray[:,3]
    bluemean = np.mean(blueTrace)
    bluestd = np.std(blueTrace)
    norm_blue = (blueTrace - bluemean) / bluestd

    traceArray[:,1] = norm_red
    traceArray[:,2] = norm_green
    traceArray[:,3] = norm_blue

    # mintime = (np.amin(traceArray, axis=0)[0])  # since the first column is the time
    # maxtime = (np.amax(traceArray,axis=0)[0]) #since the first column is the time
    # frameRate = len(traceArray) / (maxtime - mintime)
    # print "Frame rate: ", frameRate

    times = traceArray[:,0]

    # delete the time column from the trace array
    traceArray = np.delete(traceArray, 0, 1)

    #perform ICA on the three colors treating them as signals
    #the input signals - red, green and blue traces
    matrixSignals = np.matrix(traceArray)

    # #use a random normal distribution as a 3x3 mixing matrix
    # stdmean = 0.0
    # stdvar = 1.0
    # mixingMatrix = np.random.normal(stdmean,stdvar, (3,3))
    #
    # #multiply the signal and mixing matrix
    # mixedSignal = np.dot(matrixSignals, mixingMatrix.T)

    #compute ICA using FastICA from Scipy module
    ica = FastICA(n_components = 3,max_iter=1000)
    reconstructedSignal = ica.fit_transform(matrixSignals)

    #get individual components
    reconstructedComponent1 = reconstructedSignal[:,0]
    reconstructedComponent2 = reconstructedSignal[:,1]
    reconstructedComponent3 = reconstructedSignal[:,2]

    tracelength = len(traceArray)

    #extract the heart rate for individual components
    hr1 = getHeartRateForComponent(reconstructedComponent1,times,tracelength,frameRate)
    hr2 = getHeartRateForComponent(reconstructedComponent2,times,tracelength,frameRate)
    hr3 = getHeartRateForComponent(reconstructedComponent3,times,tracelength,frameRate)

    #maximum of the heart rate values are output as the correct value
    # hr = max(hr1,hr2,hr3)
    hr = hr2
    print "heart rate : ", hr
    print "c1 : ", hr1
    print "c2 : ", hr2
    print "c3 : ", hr3



    # if count % 30 == 0:
    #plot the powerspectrum plot
    # plotGraphs(f1,p1,mpidx1,f2,p2,mpidx2,f3,p3,mpidx3)

    return hr

#plot graphs
def plotGraphs(f1,p1,mpidx1,f2,p2,mpidx2,f3,p3,mpidx3):
    fig = plt.figure(1)
    plt.clf()

    fig.suptitle('Beats per minute(Frequency) vs Power', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(3,1,1)
    ax.set_title('Component 1')
    ax.set_xlabel('Beats per minute')
    ax.set_ylabel('Power')
    idx = np.argsort(f1)
    ax.plot(f1[idx],p1[idx])
    ax.plot(f1[mpidx1],p1[mpidx1], 'o')

    ax = fig.add_subplot(3,1,2)
    ax.set_title('Component 2')
    ax.set_xlabel('Beats per minute')
    ax.set_ylabel('Power')
    idx = np.argsort(f2)
    ax.plot(f2[idx], p2[idx])
    ax.plot(f2[mpidx2], p2[mpidx2], 'o')

    ax = fig.add_subplot(3,1,3)
    ax.set_title('Component 3')
    ax.set_xlabel('Beats per minute')
    ax.set_ylabel('Power')
    idx = np.argsort(f3)
    ax.plot(f3[idx], p3[idx])
    ax.plot(f3[mpidx3], p3[mpidx3], 'o')

    plt.show()


#main program start
#create a video capture object to read the video
# cap = cv2.VideoCapture('Test_Video.avi')
cap = cv2.VideoCapture('Yuan_before.avi')
# cap = cv2.VideoCapture(0)
plt.ion()

fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print "Frames per second: ", format(fps)

count = 0
bpmcount = 0
start_time = time.time()
tracesList = []
thresholdTraces = 70
while(1):
    ret ,frame = cap.read()
    if ret == True:

        count += 1

        print "Processing frame " , count
        frame, totalFaces = detectFacesInImage(frame)

        if(len(totalFaces) != 0):
            detectedFace = totalFaces[0]

            tracesList = collectDataForHeartRateEstimate(tracesList, detectedFace)

            videoWindow1 = "Detected face"
            cv2.namedWindow(videoWindow1, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(videoWindow1, detectedFace)
        else:
            #if no faces found in this frame, use the previous reading
            print "No face detected using the last reading values again"
            if(len(tracesList) > 0):
                lastReading = tracesList[-1]
                elapsed_time = time.time() - start_time
                lastReading[0] = elapsed_time
                print lastReading
                tracesList = addReadingToTraces(tracesList, lastReading)

        # check if the number of readings in traces list are more than the threshold
        if len(tracesList) >= thresholdTraces:
            if(bpmcount == 2):
                bpmcount = 0
                bpm = estimateHeartRate(tracesList,fps)
                # put the text of heart rate
                heartRate = 'Estimated heart rate : ' + str(bpm)
            else:
                bpmcount+=1

        else:
            # put the text that data is being recorded
            needed = (thresholdTraces - len(tracesList))
            heartRate = 'Collecting data - need ' + str(needed) + ' more observations'

        # show the video frame
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, heartRate, (10, 50), font, 1, (0, 0, 0))

        frame = cv2.resize(frame, (700, 500))
        videoWindow = "Heart Rate Estimation"
        cv2.namedWindow(videoWindow, cv2.WINDOW_AUTOSIZE)
        # cv2.resizeWindow(videoWindow, 1440, 810)
        cv2.imshow(videoWindow, frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()