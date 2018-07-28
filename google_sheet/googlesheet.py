# -*- coding: UTF-8 -*-

import urllib2

key = 'AIzaSyBY13d6o1xCWWNRowCuCveTbglbpQbh__Q'
sheetId = '1NwTCLZXwXdP49j9oaZymdSLzLnHTZDipqKa9_zZHDiI'
# test-420@propane-flow-211413.iam.gserviceaccount.com
range = 'Sheet1!A1:A1'

url = 'https://sheets.googleapis.com/v4/spreadsheets/' + sheetId + '/values/' + range

print url

# req = urllib2.Request(url)
# res_data = urllib2.urlopen(req)
# res = res_data.read()
# print res

