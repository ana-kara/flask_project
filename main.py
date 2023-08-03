from flask import Flask, render_template, request, url_for, redirect, url_for, flash, session
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from flask_bootstrap import Bootstrap
#from flask_mysqldb import MySQL
import mysql.connector
import os
import re
from folium.plugins import MarkerCluster
import folium
from folium import Map, Marker, Popup, Icon
from markupsafe import Markup
from PIL import Image

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# model = load_model("C:\\xampp\\htdocs\\minerals\\my_model_4.h5")
# model.load_weights("C:\\xampp\\htdocs\\minerals\\my_model_weights_4.h5")


# img_size = 224

# def predict_mineral(file_path):
#   img = Image.open(file_path)
#   img = img.resize((img_size, img_size))
#   img = np.expand_dims(img, axis=0)
#   img = img/255.0
#   prediction = model.predict(img)
#   predicted_class = np.argmax(prediction)
#   if predicted_class == 0:
#     return "Calcite"
#   elif predicted_class == 1:
#     return "Fluorite"
#   elif predicted_class == 2:
#     return "Halite"
#   elif predicted_class == 3:
#     return "Clear quartz"
#   else:
#     return "Unknown"




# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:\\xampp\\htdocs\\minerals\\my_model_4_compressed.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_size = 224

def predict_mineral(file_path):
  img = Image.open(file_path)
  img = img.resize((img_size, img_size))
  img = np.array(img).astype('float32') / 255.0
  img = np.expand_dims(img, axis=0)

  # Set the tensor to be the input for the TFLite model
  interpreter.set_tensor(input_details[0]['index'], img)

  # Run the inference
  interpreter.invoke()

  # Get the predicted class
  prediction = interpreter.get_tensor(output_details[0]['index'])
  predicted_class = np.argmax(prediction)

  if predicted_class == 0:
    return "Calcite"
  elif predicted_class == 1:
    return "Fluorite"
  elif predicted_class == 2:
    return "Halite"
  elif predicted_class == 3:
    return "Clear quartz"
  else:
    return "Unknown"








secret_key = os.urandom(24)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PASSWORD'] = 'uherqpwuykaxcxwh'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'anastasian.kivigalleria@gmail.com'
app.config['MAIL_DEFAULT_SENDER'] = 'anastasian.kivigalleria@gmail.com'


mail = Mail(app)

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])




def get_credentials():
    creds = None
    token_file = os.path.join('C:\\', 'xampp', 'htdocs', 'minerals', 'token.json')
    scope = ['https://mail.google.com/']
    flow = InstalledAppFlow.from_client_secrets_file(
        os.path.join('C:\\', 'xampp', 'htdocs', 'minerals', 'credentials.json'), scopes=scope)
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scope)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f'Please visit this URL to authorize the application: {auth_url}')
            code = input('Enter the authorization code: ')
            creds = flow.fetch_token(code=code)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    return creds





bootstrap = Bootstrap(app)
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)

def jinja2_enumerate(iterable):
    return enumerate(iterable)
app.jinja_env.filters['enumerate'] = jinja2_enumerate

def jinja2_len(iterable):
    return len(iterable)

app.jinja_env.filters['len'] = jinja2_len


conn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="minerals"
)
db_cursor = conn.cursor()




@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/logout')
def logout():
    session.clear()
    #flash('You have been logged out', 'success')
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404



@app.route("/kokoelma")
def display_table():
  # Execute a SELECT statement to retrieve the data
  cursor = conn.cursor()
  cursor.execute("SELECT id,nimi,kuva,hinta,mitat,hankittu,hankintavuosi,louhittu FROM mineraalit")
  rows = cursor.fetchall()

  # Render the HTML template with the data
  return render_template("kokoelma.html", rows=rows)


class CordForm(FlaskForm):
    lat = StringField('lat', validators=[DataRequired()])
    lng = StringField('lng', validators=[DataRequired()])
    description = StringField("desc", validators=[DataRequired()])
    submit = SubmitField('Login')

@app.route("/kokoelma/<name>", methods=["GET", "POST"])
def user_2(name):
    template_name = name
    template_path = os.path.join(r"C:\xampp\htdocs\minerals\static",template_name)

    files = os.listdir(template_path)
    items = []
    marker_data = []
    
    for file in files:
        if file.endswith('.mp4'):
            items.append({'src': os.path.join(template_path, file), 'type': 'video'})
        elif file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            items.append({'src': os.path.join(template_path, file), 'type': 'image'})
        elif file.endswith('.txt'):
            path1 = os.path.join('C:\\xampp\\htdocs\\minerals\\static', name, file)
            with open(path1, 'r') as f:
              data = f.read().split('\n\n')
            marker_data = eval(data[2])             

    cleaned_name = re.sub(r"[^A-Za-zöä]+", ' ', name) 

    map = folium.Map(location=[0,0], zoom_start=2)

    # add marker using coordinates
    markers = MarkerCluster().add_to(map)
    for one_data in eval(data[2]):
      folium.Marker(location=one_data[0:2], popup=one_data[4]).add_to(markers)

    map_html = map._repr_html_()
    form = CordForm()
    if request.method == "POST" and form.validate_on_submit():
        lat = form.lat.data
        lng = form.lng.data
        description = form.description.data

        if lat and lng and description:
            marker_data.append([float(lat), float(lng),0,0,description])
            with open(path1, "w",encoding='utf-8') as f:
                f.write(data[0] + "\n\n" + data[1] + "\n\n" + repr(marker_data))

            folium.Marker(location=[float(lat), float(lng)], popup=description).add_to(markers)
        return redirect('/kokoelma/{}'.format(name))

    return render_template('base_collection.html', name=name, title=cleaned_name, items=items, content1=data, map_html=Markup(map_html),form=form)
   

class PredictionForm(FlaskForm):
    image = FileField('image', validators=[FileRequired(), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])


@app.before_request
def csrf_protect():
    if request.method == "POST":
        csrf.protect()

@app.route('/classify')
def home():
    form = PredictionForm()
    return render_template('classify.html',form=form)

@app.route('/predict', methods=['POST'])
@csrf.exempt
def predict():
    form = PredictionForm()
    if form.validate_on_submit():
        file = request.files['image']
        file_path = 'C:\\xampp\\htdocs\\minerals\\static\\user_pic\\' + file.filename
        file.save(file_path)
        prediction = predict_mineral(file_path)
        return render_template('result.html', prediction=prediction, file_path=file_path)
    return render_template('classify.html', form=form)




@app.route('/register', methods=['GET', 'POST'])
def register():
  if request.method == 'POST':
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    
    db_cursor.execute('SELECT COUNT(*) FROM new_users WHERE username=%s', (username,))
    if db_cursor.fetchone()[0] > 0:
      flash('Username already taken','warning')
      return redirect('/register')

    db_cursor.execute('SELECT COUNT(*) FROM new_users WHERE email=%s', (email,))
    if db_cursor.fetchone()[0] > 0:
      flash('Email address already registered','warning')
      return redirect('/register')

    agree = request.form.get('agree')
    if agree != 'yes':
       flash('You must agree to the terms before registering.','warning')
       return redirect('/register')

    token = serializer.dumps(email, salt='email-verification')

    # Send a verification email to the user
    verification_link = url_for('verify_email', token=token, _external=True)
    msg = Message('Verify your email', recipients=[email])
    msg.body = f'Click this link to verify your email: {verification_link}'
    #with mail.connect() as conn:
     #   conn.credentials = get_credentials()
      #  conn.send(msg)
    mail.send(msg)

# Add the user to the database with an unverified email address
    db_cursor.execute('INSERT INTO new_users (username, email, password_hash, verified) VALUES (%s, %s, %s, %s)',
    (username, email, generate_password_hash(password), False))
    conn.commit()
    flash('Registration successful. Please check your email for a verification link.', 'success')
    return redirect('/register')

  return render_template('registration.html')



@app.route('/verify_email/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-verification', max_age=86400)

        db_cursor.execute("SELECT * FROM new_users WHERE email=%s", (email,))
        result = db_cursor.fetchone()
    
        if result:
          db_cursor.execute("UPDATE new_users SET verified = TRUE WHERE email=%s", (email,))
          conn.commit()
          flash('Email verification successful. You can now sign in.','success')
          return redirect('/login')
        else:
          flash('Email verification failed. Please try again.','warning')
          return redirect('/login')

    except SignatureExpired:
        return 'Verification link expired'


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST'and form.validate_on_submit():
        username = request.form['username']
        password = request.form['password']

        db_cursor.execute("SELECT * FROM new_users WHERE username=%s", (username,))
        result = db_cursor.fetchone()

        if result is None:
        # username not found in database
          flash('Invalid username or password','warning')
          return redirect('/login')

        else:
        # username found in database, check if password is correct
          stored_password = result[3]
          verified_true = result[4]
          if check_password_hash(stored_password, password):
            if verified_true == True:
               
            # password is correct, store username in session and redirect to home page
              session['username'] = username
              return redirect('/')
            
            # QaLm6dWZGJ4DwAq!
            # QaLm6dWZGJ4DwAq!!!!!!!22
            else:
              flash('Validate your email first','warning')
              return redirect('/login')
          else:
            # password is incorrect
            flash('Invalid username or password','warning')
            return redirect('/login')
    else:
       return render_template('login.html',form=form)









if __name__ == '__main__':
  app.run(debug=True)
