from RecommendMovie import *
from pre_Process_Data import *
from flask import Flask, render_template, request



app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/register')
def start():
    return render_template("index.html")


@app.route('/search', methods=["GET", "POST"])
def register():
    if request.form.get("user_name") == "" or request.form.get("password") == "":
        return render_template("register_page.html")
    else:
        registr_info = ''
        registr_info += request.form.get("user_name", '') + '\t'
        registr_info += request.form.get("email", '') + '\t'
        registr_info += request.form.get("password", '') + '\t'
        registr_info += request.form.get("secret_word", '') + '\t'
        registr_info += '\n'
        print(registr_info)
        with open('registration.csv', 'a') as registr_file:
            registr_file.write(registr_info)
        registr_file.close()
    return render_template('search_page.html')


@app.route('/movie_search_res', methods=["GET", "POST"])
def repr_movie_search_res():
    #if request.form.get("search_str") == '':
        #return render_template('search_page.html')
    MovieID = request.form.get("Movie_ID")
    UserID = request.form.get("User_ID")
    #if not movie_title:
        #return render_template('search_page.html')
    #movie_title.strip()
    User_Movie=getUserMovie(int(UserID))
    movie_chosen,movie_Rec1=recommend_same_type_movie(int(MovieID),20)
    movie_Rec2=recommend_your_favorite_movie(int(UserID),10)
    movie_Rec3=recommend_other_favorite_movie(int(MovieID),20)
    print('####')
    print(User_Movie)
    print('####')
    #print(movie_Rec1)
    print('####')
    #print(movie_Rec2)
    #print('####')
    #print(movie_Rec3)
    #return render_template('index.html')
    return render_template('index.html', movie = movie_chosen,recommend1=movie_Rec1,recommend2=movie_Rec2,recommend3=movie_Rec3,userMovie=User_Movie,userID=int(UserID))

if __name__ == '__main__':
   app.run(debug = True)
