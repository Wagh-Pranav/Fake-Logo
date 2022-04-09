from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib import messages
from django.http import HttpResponseForbidden
import random
import cv2
import os
import numpy
import math
from pathlib import Path
import numpy as np
from account.models import Profile

from detection.models import UploadImage

BASE_DIR = Path(__file__).resolve().parent.parent

num_images = 2700


# Create your views here.
def augment(request):
    if(request.session.has_key('account_id')):
        if request.session['account_role'] == 1:
            content = {}
            content['title'] = 'Augment Logo Data & Test Probe Data'
            if request.method == "POST":
                for i in range(0,num_images):
                    f=random.choice(os.listdir(str(BASE_DIR) + '/logo_data/logos/'))
                    print(f)
                    logoI = cv2.imread(str(BASE_DIR) + '/logo_data/logos/'+f)
                    logo = cv2.cvtColor(logoI, cv2.COLOR_BGR2GRAY)   
                    
                    scale = 1;#random.randint(0,5)
                    if scale!=0 and ((logo.shape[0]/scale)>10):  
                        M, out_of_bounds = make_affine_transform(
                                            from_shape=logo.shape,
                                            to_shape=logo.shape,
                                            min_scale=1,
                                            max_scale=1,
                                            rotation_variation=1.0,
                                            scale_variation=1.5,
                                            translation_variation=1.2)   
                                            
                        im = cv2.warpAffine(logo,M,logo.shape,255)

                        print (str(BASE_DIR) + '/augmentedData/'+str(i)+'_'+f)
                        cv2.imwrite(str(BASE_DIR) + '/augmentedData/'+str(i)+'_'+f, im)
                    else:
                        print (str(BASE_DIR) + '/augmentedData/'+str(i)+'_'+f)
                        cv2.imwrite(str(BASE_DIR) + '/augmentedData/'+str(i)+'_'+f, im)
                messages.success(request, 'Logos augmented')
            return render(request, 'admin/augment.html', content)
        else:
            return HttpResponseForbidden()
    else:
        return HttpResponseRedirect(reverse('account-login'))

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    scale *= numpy.min(to_size / skewed_size)

    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def test(request):
    if(request.session.has_key('account_id')):
        if request.session['account_role'] == 1:
            content = {}
            content['title'] = 'Augment Logo Data & Test Probe Data'
            content['accuracy'] = '0'
            if request.method == "POST":
                detector = cv2.xfeatures2d.SIFT_create(60)
                uarray=[4]              #can be used to tune hyper parameters
                tharray = [0.65]
                for u in uarray:
                    for th in tharray:
                        i=0
                        kps2 =[]
                        descs2 =[]
                        brand =[]
                        MAX_LIMIT =1
                        files=os.listdir(str(BASE_DIR) + '/logo_data/logos/')
                        # store the descriptors for all the logos
                        for f in files:
                                logoI = cv2.imread(str(BASE_DIR) + '/logo_data/logos/'+f)
                                logo = cv2.cvtColor(logoI, cv2.COLOR_BGR2GRAY)
                                (kps, descs) = detector.detectAndCompute(logo, None)
                                kps2.append(kps)
                                descs2.append(descs)
                                brand.append(f.split('.')[0])        
                                
                        files=os.listdir(str(BASE_DIR) + '/augmentedData/')        
                        for f in files:             # add synthetic images to the logos
                                logoI = cv2.imread(str(BASE_DIR) + '/augmentedData/'+f)
                                logo = cv2.cvtColor(logoI, cv2.COLOR_BGR2GRAY)
                                (kps, descs) = detector.detectAndCompute(logo, None)
                                kps2.append(kps)
                                descs2.append(descs)
                                brand.append(f.split('_')[1].split('.')[0])
                        
                        # read probes.txt file
                        with open(str(BASE_DIR) + '/logo_data/probes.txt') as f:
                            lines = f.readlines()
                        detector = cv2.xfeatures2d.SIFT_create() 
                        # process the probes.txt file line by line
                        actualBrand=[]
                        logoMatchList=[]
                        cnt = 0
                        for line in lines:
                            hashMap={}
                            logoMatch=[]
                            filename = line.split("\t")[0]
                            brandname = line.split("\t")[1].strip()
                            probeI = cv2.imread(str(BASE_DIR) + '/logo_data/probes/' + filename)
                            probe = cv2.cvtColor(probeI, cv2.COLOR_BGR2GRAY)
                            detector = cv2.xfeatures2d.SIFT_create()
                            (kps1, descs1) = detector.detectAndCompute(probe, None)
                            i=0
                            for d,kp in zip(descs2,kps2):
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(d,descs1,k=3)
                                good = []
                                
                                for m,n,k in matches:
                                    if m.distance < th*n.distance:# and m.distance < 0.5*k.distance:
                                        good.append(m)
                                good = sorted(good, key=lambda val: val.distance)
                        
                                if len(good)>u:
                                    if hashMap.get(brand[i]) is None:
                                        hashMap[brand[i]]=1
                                    else:
                                        break
                                    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                                    dst_pts = np.float32([kps1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                                    # print(src_pts)
                                    # print(dst_pts)
                                                    
                                    #get homography to draw the bounding box
                                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,8.0)
                                    # print(M)
                                    if M is not None:           
                                        matchesMask = mask.ravel().tolist()
                                        logoMatch.append(brand[i])
                                        h,w = logo.shape
                                        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                                        dst = cv2.perspectiveTransform(pts,M)
                        
                                        # probe = cv2.polylines(probe,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
                                        # cv2.imwrite('matchResults/'+filename+'_'+brand[i]+'_out.png', probe)
                                        print(brand[i])
                                    else:
                                        matchesMask = None
                                i=i+1
                            logoMatchList.append(logoMatch)
                            actualBrand.append(brandname)
                        
                        # computing accuracy
                        tp=0
                        tn=0
                        fp=0
                        fn=0    
                        i=0
                        
                        for brands in logoMatchList:
                            for brand in brands:
                                if(brand ==actualBrand[i]):
                                    tp=tp+1
                                else:
                                    fp=fp+1
                            i=i+1
                        i=0
                        
                        for brands in logoMatchList:
                            if (brands==[]):
                                if (actualBrand[i]=='none'):
                                    tn=tn+1
                                else:
                                    fn=fn+1
                            i=i+1
                        content['accuracy'] = "Accuracy is " + str((float)(tp+tn)/(float)(tp+tn+fp+fn))
                messages.success(request, 'Probe Test Done.')
            return render(request, 'admin/augment.html', content)
        else:
            return HttpResponseForbidden()
    else:
        return HttpResponseRedirect(reverse('account-login'))

# Detect
def upload(request):
    if(request.session.has_key('account_id')):
        if request.session['account_role'] == 2:
            content = {}
            content['title'] = 'Upload Image'
            if request.method == "POST":
                image = request.FILES['image']
                upload_data = UploadImage()
                upload_data.image = image
                upload_data.profile = Profile.objects.get(pk = int(request.session['account_id']))
                upload_data.save()
                last_id = UploadImage.objects.filter(profile_id = int(request.session['account_id'])).last()
                return HttpResponseRedirect(reverse('detect', kwargs={'pk' : last_id.id}))
            return render(request, 'home/upload.html', content)
        else:
            return HttpResponseForbidden()
    else:
        return HttpResponseRedirect(reverse('account-login'))

def detect(request, pk):
    if(request.session.has_key('account_id')):
        if request.session['account_role'] == 2:
            content = {}
            content['title'] = 'Results'
            image_data = UploadImage.objects.get(pk=pk)
            content['image_data'] = image_data
            print(str(BASE_DIR) + '/media/' + str(image_data.image))
            content['result'] = ''
            detector = cv2.xfeatures2d.SIFT_create(60)
            uarray=[4]              #can be used to tune hyper parameters
            tharray = [0.65]
            for u in uarray:
                for th in tharray:
                    i=0
                    kps2 =[]
                    descs2 =[]
                    brand =[]
                    MAX_LIMIT =1
                    files=os.listdir(str(BASE_DIR) + '/logo_data/logos/')
                    # store the descriptors for all the logos
                    for f in files:
                        logoI = cv2.imread(str(BASE_DIR) + '/logo_data/logos/'+f)
                        logo = cv2.cvtColor(logoI, cv2.COLOR_BGR2GRAY)
                        (kps, descs) = detector.detectAndCompute(logo, None)
                        kps2.append(kps)
                        descs2.append(descs)
                        brand.append(f.split('.')[0])        
                            
                    files=os.listdir(str(BASE_DIR) + '/augmentedData/')        
                    for f in files:             # add synthetic images to the logos
                        logoI = cv2.imread(str(BASE_DIR) + '/augmentedData/'+f)
                        logo = cv2.cvtColor(logoI, cv2.COLOR_BGR2GRAY)
                        (kps, descs) = detector.detectAndCompute(logo, None)
                        kps2.append(kps)
                        descs2.append(descs)
                        brand.append(f.split('_')[1].split('.')[0])
                    
                    # read probes.txt file
                    with open(str(BASE_DIR) + '/logo_data/probes.txt') as f:
                        lines = f.readlines()
                    detector = cv2.xfeatures2d.SIFT_create() 
                    # process the probes.txt file line by line
                    actualBrand=[]
                    logoMatchList=[]
                    # for line in lines:
                    hashMap={}
                    logoMatch=[]
                    # filename = line.split("\t")[0]
                    # brandname = line.split("\t")[1].strip()
                    filename = image_data.image
                    probeI = cv2.imread(str(BASE_DIR) + '/media/' + str(filename))
                    probe = cv2.cvtColor(probeI, cv2.COLOR_BGR2GRAY)
                    detector = cv2.xfeatures2d.SIFT_create()
                    (kps1, descs1) = detector.detectAndCompute(probe, None)
                    i=0
                    for d,kp in zip(descs2,kps2):
                        bf = cv2.BFMatcher()
                        matches = bf.knnMatch(d,descs1,k=3)
                        good = []
                        
                        for m,n,k in matches:
                            if m.distance < th*n.distance:# and m.distance < 0.5*k.distance:
                                good.append(m)
                        good = sorted(good, key=lambda val: val.distance)
                
                        if len(good)>u:
                            if hashMap.get(brand[i]) is None:
                                hashMap[brand[i]]=1
                            else:
                                break
                            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                            dst_pts = np.float32([kps1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                            # print(src_pts)
                            # print(dst_pts)
                                            
                            #get homography to draw the bounding box
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,8.0)
                            # print(M)
                            if M is not None:           
                                matchesMask = mask.ravel().tolist()
                                logoMatch.append(brand[i])
                                h,w = logo.shape
                                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                                dst = cv2.perspectiveTransform(pts,M)
                
                                # probe = cv2.polylines(probe,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
                                # cv2.imwrite('matchResults/'+filename+'_'+brand[i]+'_out.png', probe)
                                brand_name_is = brand[i]
                                content['result'] = brand_name_is.title()
                            else:
                                content['result'] = 'No data found'
                                matchesMask = None
                        i=i+1
                    logoMatchList.append(logoMatch)
                    # actualBrand.append(brandname)
            # messages.success(request, 'Probe Test Done.')
            return render(request, 'home/detect.html', content)
        else:
            return HttpResponseForbidden()
    else:
        return HttpResponseRedirect(reverse('account-login'))