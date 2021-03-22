from flask import Blueprint, render_template, request, flash, redirect, url_for
from random import randint
import requests
import os


home = Blueprint('home', __name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './../static/'

region = "westus" #For example, "westus"
api_key = "6c8a89a3dbc142e592ae18b97061ae27"



@home.route('/home')
@home.route('/', methods=['GET'])
def homepage():
    return render_template('home.html', title='Home')


@home.route('/about')
def about():
    return render_template('about.html', title='About')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
# @app.route('/')
# def upload_form():
#     return render_template('upload.html')


@home.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        print(file.filename)
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        dirname = os.path.dirname(__file__)
        filename1 = os.path.join(dirname, '../static/')
        file.save(os.path.join(filename1, filename))
        print(os.path.join(filename1, filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        with open(os.path.join(filename1, filename), 'rb') as f:
            data = f.read()
        
        prediction,confidence = getTBPrediction(data)
        return render_template('home.html', filename=filename, prediction = prediction,confidence = "{:.2f}".format(confidence))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

    
    
def getTBPrediction(data):
    # Set request headers
    headers = dict()
    headers['Prediction-Key'] = api_key
    headers['Content-Type'] = 'application/octet-stream'

    # Set request querystring parameters
    params = {'visualFeatures': 'Color,Categories,Tags,Description,ImageType,Faces,Adult'}

    # Make request and process response
    response = requests.request('post',"https://southcentralus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/67aef175-4b37-4a63-95cf-7e309123df36/classify/iterations/Iteration1/image", data=data, headers=headers, params=params)

    if response.status_code == 200 or response.status_code == 201:

        if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
            result = None
        elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
            if 'application/json' in response.headers['content-type'].lower():
                result = response.json() if response.content else None
            elif 'image' in response.headers['content-type'].lower():
                result = response.content
            flag = 0
            if result['predictions'][0]['probability'] > result['predictions'][1]['probability']:
                tagname = "TB"
                flag =0
            else:
                tagname = "Normal"
                flag = 1 
           
            confidence = result['predictions'][0]['probability']
            print(result)
            return result['predictions'][1]['tagName'], confidence
    else:
        print("Error code: %d" % response.status_code)
        print("Message: %s" % response.json())
        
        
@home.route('/display/<filename>')
def display_image(filename):
#     print('display_image filename: ' + filename + url_for('static', filename='/' + filename))
#     print('../static/'+ filename)
    dirname = os.path.dirname(__file__)
    filename1 = os.path.join(dirname, '../static/')
#     file.save(os.path.join(filename1, filename))
    with open(os.path.join(filename1, filename), 'rb') as f:
        print("hello")
        data = f.read()
        
    getTBPrediction(data)
    return redirect(url_for('static', filename='/' + filename), code=301)



