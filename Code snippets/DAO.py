from dbconnect import connection

import mysql.connector
class DAO:
   #function to login 
    def login(username, password):
        try:
            c, conn = connection()
            c.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username,password,))
            user = c.fetchone()
            if not user:
                c.close()
                conn.close()
                return False
            else:
                c.close()
                conn.close()
                return True
        except mysql.connector.Error as error:
            conn.rollback()
            print("Failed searching username and password in database {}".format(error))
 
    #funciton to insert new user into database
    def register(username, password, email, firstname, lastname):
        try:
            c,conn = connection()
            sql = "INSERT INTO users (username, password, email, firstname,lastname) VALUES (%s, %s, %s, %s, %s)"
            val = (username, password, email, firstname, lastname)
            c.execute(sql, val)
            conn.commit()
        except mysql.connector.Error as error :
            conn.rollback()
            print("Failed inserting data into MySQL table {}".format(error))
        finally:
            c.close()
            conn.close()
            return True
    
    #insert into images table value of an image, include imagename, objects under JSON type, photo data under BLOB type
    def insert_image(filename, data, objects):
        try:
            c,conn = connection()
            sql = "INSERT INTO images (imagename, photo,objects) VALUES (%s, %s, %s)"
            val = (filename,data,objects)
            c.execute(sql,val)
            conn.commit()
            c.close()
            conn.close()
        except mysql.connector.Error as error :
            conn.rollback()
            print("Failed searching data in MySQL table {}".format(error))

    
    #function to update images table by set newphoto field with values of detected image 
    def update_image(data,filename):
        try:
            c,conn = connection()
            sql = "UPDATE images SET newphoto = %s WHERE imagename = %s"
            val = (data, filename)
            c.execute(sql, val)
            conn.commit()
            c.close()
            conn.close()
        except mysql.connector.Error as error :
            conn.rollback()
            print("Failed searching data in MySQL table {}".format(error))
    #insert into objects table all objects' properties of an image
    def insert_objects(imagename,objectname,score,x,y,width,height):
        try:
            c,conn = connection()
            sql = "INSERT INTO objects (imagename,class_name,score,x,y,width,height) VALUES (%s,%s,%s,%s,%s,%s,%s)"
            val = (imagename,objectname,score,x,y,width,height)
            c.execute(sql, val)
            conn.commit()
        except mysql.connector.Error as error :
            conn.rollback()
            print("Failed inserting data into MySQL table {}".format(error))
        finally:
            c.close()
            conn.close()
    #function retrieve an original photo from images table by image name
    def retrieve_photo(imagename):
        try:
            c,conn = connection()
            sql = "SELECT photo from images where imagename = %s"
            val=(imagename,)
            c.execute(sql, val)
            record = c.fetchone()
            if(c.rowcount>0):
                c.close()
                conn.close()
                return record
            else:
                return 0
        except mysql.connector.Error as error :
            conn.rollback()
            print("Failed searching data int MySQL table {}".format(error))

    #function search image name from objects table by name of an object
    def search_image_name_from_object(name):
        try:
            c,conn = connection()
            sql = "SELECT imagename from objects WHERE class_name = %s"
            val = (name,)
            c.execute(sql,val)
            record = c.fetchall()
            if(c.rowcount>0):
                c.close()
                conn.close()
                return record
            else:
                return 0
        except mysql.connector.Error as error:
            conn.rollback()
            print("Failed searching data int MySQL table {}".format(error))
    
   
        
        
        
