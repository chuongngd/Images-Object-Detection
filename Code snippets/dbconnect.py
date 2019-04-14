
import mysql.connector

#function to connect database
def connection():
	try:
		conn = mysql.connector.connect(user='root',password='root',host='127.0.0.1',db='flask',connect_timeout=1000)
		c = conn.cursor(buffered=True)
		return c, conn
	except mysql.connector.Error as error:
		print("Failed to connect to mysql {}".format(error))
		conn.reconnect(attemps=1, delay=0)
		