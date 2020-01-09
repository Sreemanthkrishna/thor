import os
import datetime
import hashlib
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import list_users, verify, delete_user_from_db, add_user
from database import read_note_from_db, write_note_into_db, delete_note_from_db, match_user_id_with_note_id
from database import image_upload_record, list_images_for_user, match_user_id_with_image_uid, delete_image_from_db
from werkzeug.utils import secure_filename
from flask import g
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import math
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve





app = Flask(__name__)
app.config.from_object('config')



@app.errorhandler(401)
def FUN_401(error):
    return render_template("page_401.html"), 401

@app.errorhandler(403)
def FUN_403(error):
    return render_template("page_403.html"), 403

@app.errorhandler(404)
def FUN_404(error):
    return render_template("page_404.html"), 404

@app.errorhandler(405)
def FUN_405(error):
    return render_template("page_405.html"), 405

@app.errorhandler(413)
def FUN_413(error):
    return render_template("page_413.html"), 413





@app.route("/")
def FUN_root():
    return render_template("index.html")

@app.route("/public/")
def FUN_public():
    return render_template("public_page.html")

@app.route("/private/")
def FUN_private():
    if "current_user" in session.keys():
        notes_list = read_note_from_db(session['current_user'])
        notes_table = zip([x[0] for x in notes_list],\
                          [x[1] for x in notes_list],\
                          [x[2] for x in notes_list],\
                          ["/delete_note/" + x[0] for x in notes_list])

        images_list = list_images_for_user(session['current_user'])
        images_table = zip([x[0] for x in images_list],\
                          [x[1] for x in images_list],\
                          [x[2] for x in images_list],\
                          ["/delete_image/" + x[0] for x in images_list])

        return render_template("private_page.html", notes = notes_table, images = images_table)
    else:
        return abort(401)

@app.route("/admin/")
def FUN_admin():
    if session.get("current_user", None) == "ADMIN":
        user_list = list_users()
        user_table = zip(range(1, len(user_list)+1),\
                        user_list,\
                        [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
        return render_template("admin.html", users = user_table)
    else:
        return abort(401)






@app.route("/write_note", methods = ["POST"])
def FUN_write_note():
    text_to_write = request.form.get("text_note_to_take")
    write_note_into_db(session['current_user'], text_to_write)

    return(redirect(url_for("FUN_private")))

@app.route("/delete_note/<note_id>", methods = ["GET"])
def FUN_delete_note(note_id):
    if session.get("current_user", None) == match_user_id_with_note_id(note_id): # Ensure the current user is NOT operating on other users' note.
        delete_note_from_db(note_id)
    else:
        return abort(401)
    return(redirect(url_for("FUN_private")))


# Reference: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','xls','arff','tsv','csv','xlsx'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return(redirect(url_for("FUN_private")))
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return(redirect(url_for("FUN_private")))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_time = str(datetime.datetime.now())
            image_uid = hashlib.sha1((upload_time + filename).encode()).hexdigest()
            # Save the image into UPLOAD_FOLDER
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            # Record this uploading in database
            image_upload_record(image_uid, session['current_user'], filename, upload_time)
            return(redirect(url_for("FUN_private")))

    return(redirect(url_for("FUN_private")))
    
    
@app.route("/display_data", methods=['POST'])
def FUN_display_data(): 
    if request.method=='POST':
        if 'analyzedata' not in request.files:
            flash('No file part')
            return(redirect(url_for("FUN_private")))
        analyzedata = request.files['analyzedata']
           
        df = pd.read_csv(analyzedata)
        
        
        # above line will be different depending on where you saved your data, and your file name
        df.head(5)
        return render_template('private_page.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
        return(redirect(url_for("FUN_private")))

@app.route("/analyze_data",methods=['POST'])
def FUN_analyze_data():
    if request.method=='POST':
        if 'analyzedata' not in request.files:
            flash('No file part')
            return(redirect(url_for("FUN_private")))
        analyzedata = request.files['analyzedata']
        
        if analyzedata.filename == '':
            flash('No selected file')
            return(redirect(url_for("FUN_private")))
        
        data=pd.read_csv(analyzedata)
        data['text']=data.text.astype(str)
        data.head()
        data['length'] = data['text'].apply(len)
        data_classes = data[(data['stars']==1) | (data['stars']==3)| (data['stars']==5)]
        data_classes.head()
        print("class shape is",data_classes.shape)
        x = data_classes['text']
        y = data_classes['stars']
        print(x.head())
        print(y.head())
        
        def text_process(text):
            nopunc = [char for char in text if char not in string.punctuation]
            nopunc = ''.join(nopunc)
            return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        global vocab
        vocab = CountVectorizer(analyzer=text_process).fit(x)
        print(len(vocab.vocabulary_))
        
        

        x = vocab.transform(x)
        #Shape of the matrix:
        print("Shape of the sparse matrix: ", x.shape)
        #Non-zero occurences:
        print("Non-Zero occurences: ",x.nnz)
        density = (x.nnz/(x.shape[0]*x.shape[1]))*100
        print("Density of the matrix = ",density)
        
        # SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
        from sklearn.naive_bayes import MultinomialNB
        global mnb
        mnb = MultinomialNB()
        
        mnb.fit(x_train,y_train)
        predmnb = mnb.predict(x_test)
        print("Confusion Matrix for Multinomial Naive Bayes:")
        print(confusion_matrix(y_test,predmnb))
        print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
        
        print("Classification Report:/n",classification_report(y_test,predmnb))
        pr = data['text'][31]
        print(pr)
        print("Actual Rating: ",data['stars'][31])
        pr_t = vocab.transform([pr])
        
        print("Predicted Rating:")
        predicted=mnb.predict(pr_t)[0]
        print(predicted)
        
        # Dump the trained decision tree classifier with Pickle
        #warning
        #donot uncomment until you need to train the model
        
        
        
        mnbmodel = 'mnbmodel.pkl'
        # Open the file to save as pkl file
        mnbmodel = open(mnbmodel, 'wb')
        pickle.dump(mnb, mnbmodel)
        # Close the pickle instances
        mnbmodel.close()
        
        
        
        return(redirect(url_for("FUN_private"))) 


    
    
@app.route("/predict_data",methods=["POST"])
def FUN_predict_data():
    
    
    if request.method=='POST':
        if 'predictdata' not in request.files:
            flash('No file part')
            return(redirect(url_for("FUN_private")))
        predictdata = request.files['predictdata']
        if predictdata.filename == '':
            flash('No selected file')
            return(redirect(url_for("FUN_private")))
        
        #pkl unload
        mnbfrompkl = joblib.load("mnbmodel.pkl")

        pdata=pd.read_csv(predictdata)
        pdata['text']=pdata.text.astype(str)
        data_classes = pdata[(pdata['stars']==1) | (pdata['stars']==3)| (pdata['stars']==5)]
        x = data_classes['text']   
        
        
        def text_process(text):
            nopunc = [char for char in text if char not in string.punctuation]
            nopunc = ''.join(nopunc)
            return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        global vocab
        vocab = CountVectorizer(analyzer=text_process).fit(x)
        
        
        r = len(pdata)
        i=0 
        j=0
        p=[]
        for i in range(0,r):
            pr = pdata['text'][j]
            print(pr)
            print("Actual Rating: ",pdata['stars'][j])
            
            #errorerror
            pr_t = vocab.transform([pr])
            print("Predicted Rating:")
            predicted=mnbfrompkl.predict(pr_t)[0]
            print(predicted)
            j=j+1
            p.append(predicted)
                  
            predicted=0
            
        print(*p)
        
        pdata['predicted']=p
        pdata.to_csv(r'image_pool\recent.csv')
        return render_template('public_page.html', tables=[pdata.to_html(classes='data')], titles=pdata.columns.values)
        
        return(redirect(url_for("FUN_private")))    
                        
        
@app.route("/view_products",methods=["POST","GET"])
def FUN_view_products(): 
   
    #predictdata = request.files['predictdata']
       
    
    pudata=pd.read_csv('image_pool/recent.csv')
    pudata['name']=pudata.name.astype(str)
    
    productname=pudata['name'].unique()
    #print(productname)
    products=productname.tolist()
    print(*products,sep='\n')
    return render_template('public_page.html', products=products)
        
        
@app.route("/predict_users",methods=["POST"])
def FUN_predict_users():
    """if request.method=='POST':
        if 'predictdata' not in request.files:
            flash('No file part')
            return(redirect(url_for("FUN_private")))
        predictdata = request.files['predictdata']
        if predictdata.filename == '':
            flash('No selected file')
            return(redirect(url_for("FUN_private")))"""
    product = request.form.get('products')
    print(product)
    #pudata=pd.read_csv(predictdata)
    puadata=pd.read_csv('image_pool/recent.csv')
    puadata['name']=puadata.name.astype(str)
    
    productids=puadata['name'].unique()
    print(productids)
    
    pudata=puadata[puadata['name'] == product]
    
    l20=pudata[pudata['age']<20]
    
    l4=pudata[pudata['age']<40]
    
    l40=l4[l4['age']>20]
    
    l6=pudata[pudata['age']<60]
    
    l60=l6[l6['age']>40]
    
    l8=pudata[pudata['age']<80]
    
    l80=l8[l8['age']>60]
    
    
    l20m=l20[l20.sex == 1]
    l20f=l20[l20.sex == 0]
    l40m=l40[l40.sex == 1]
    l40f=l40[l40.sex == 0]
    l60m=l60[l60.sex == 1]
    l60f=l60[l60.sex == 0]
    l80m=l80[l80.sex == 1]
    l80f=l80[l80.sex == 0]
    
    
    l20mp=l20m[l20m.predicted > 2]
    l20fp=l20f[l20f.predicted > 2]
    l40mp=l40m[l40m.predicted > 2]
    l40fp=l40f[l40f.predicted > 2]
    l60mp=l60m[l60m.predicted > 2]
    l60fp=l60f[l60f.predicted > 2]
    l80mp=l80m[l80m.predicted > 2]
    l80fp=l80f[l80f.predicted > 2]
    
    
    #list for future reference
    """pl20m=l20m['predicted'].tolist()
    pl20f=l20f['predicted'].tolist()
    pl40m=l40m['predicted'].tolist()
    pl40f=l40f['predicted'].tolist()
    pl60m=l60m['predicted'].tolist()
    pl60f=l60f['predicted'].tolist()
    pl80m=l80m['predicted'].tolist()
    pl80f=l80f['predicted'].tolist()"""
    
    
    male20=len(l20mp)
    female20=len(l20fp)
    male40=len(l40mp)
    female40=len(l40fp)
    male60=len(l60mp)
    female60=len(l60fp)
    male80=len(l80mp)
    female80=len(l80fp)
    
    
    print(male20)
    print(female20)
    print(male40)
    print(female40)
    print(male60)
    print(female60)
    print(male80)
    print(female80)
    

    #p=0
    #i=0
    
    #while i < s:
    #  if(pl20m[i]>=3):
           #p=p+pl20m[i]
           #print(p)
    #      i=i+1
    #print(p)
    #print(i)
    ob=('m20','f20','m40','f40','m60','f60','m80','f80')
    objects = ('Under 20 Male', 'Under 20 Female', 'Under 40 Male', 'Under 40 Female','Under 60 Male', 'Under 60 Female','Under 80 Male', 'Under 80 Female')
    y_pos = np.arange(len(objects))
    performance = [male20,female20,male40,female40,male60,female60,male80,female80]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.xticks(y_pos,ob)
    plt.legend()
    plt.ylabel('Likelyness')
    plt.title('Best suited user for-'+product)
    plt.savefig('image_pool/new_plot.png')
    plt.show()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'new_plot.png')
    return render_template("plot.html", user_image = full_filename)
    return render_template('plot.html')
    
     
    
    
    #return render_template('public_page.html', your_list=your_list)
    return(redirect(url_for("FUN_private")))
    
    
    





@app.route("/delete_image/<image_uid>", methods = ["GET"])
def FUN_delete_image(image_uid):
    if session.get("current_user", None) == match_user_id_with_image_uid(image_uid): # Ensure the current user is NOT operating on other users' note.
        # delete the corresponding record in database
        delete_image_from_db(image_uid)
        # delete the corresponding image file from image pool
        image_to_delete_from_pool = [y for y in [x for x in os.listdir(app.config['UPLOAD_FOLDER'])] if y.split("-", 1)[0] == image_uid][0]
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete_from_pool))
    else:
        return abort(401)
    return(redirect(url_for("FUN_private")))






@app.route("/login", methods = ["POST"])
def FUN_login():
    id_submitted = request.form.get("id").upper()
    if (id_submitted in list_users()) and verify(id_submitted, request.form.get("pw")):
        session['current_user'] = id_submitted
    
    return(redirect(url_for("FUN_root")))

@app.route("/logout/")
def FUN_logout():
    session.pop("current_user", None)
    return(redirect(url_for("FUN_root")))

@app.route("/delete_user/<id>/", methods = ['GET'])
def FUN_delete_user(id):
    if session.get("current_user", None) == "ADMIN":
        if id == "ADMIN": # ADMIN account can't be deleted.
            return abort(403)

        # [1] Delete this user's images in image pool
        images_to_remove = [x[0] for x in list_images_for_user(id)]
        for f in images_to_remove:
            image_to_delete_from_pool = [y for y in [x for x in os.listdir(app.config['UPLOAD_FOLDER'])] if y.split("-", 1)[0] == f][0]
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete_from_pool))
        # [2] Delele the records in database files
        delete_user_from_db(id)
        return(redirect(url_for("FUN_admin")))
    else:
        return abort(401)

@app.route("/add_user", methods = ["POST"])
def FUN_add_user():
    if session.get("current_user", None) == "ADMIN": # only Admin should be able to add user.
        # before we add the user, we need to ensure this is doesn't exsit in database. We also need to ensure the id is valid.
        if request.form.get('id').upper() in list_users():
            user_list = list_users()
            user_table = zip(range(1, len(user_list)+1),\
                            user_list,\
                            [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
            return(render_template("admin.html", id_to_add_is_duplicated = True, users = user_table))
        if " " in request.form.get('id') or "'" in request.form.get('id'):
            user_list = list_users()
            user_table = zip(range(1, len(user_list)+1),\
                            user_list,\
                            [x + y for x,y in zip(["/delete_user/"] * len(user_list), user_list)])
            return(render_template("admin.html", id_to_add_is_invalid = True, users = user_table))
        else:
            add_user(request.form.get('id'), request.form.get('pw'))
            return(redirect(url_for("FUN_admin")))
    else:
        return abort(401)
    
    

if __name__ == "__main__":
    app.run()
