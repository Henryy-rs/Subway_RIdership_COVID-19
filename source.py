import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Data:

    def __init__(self, date_strat, date_end):

        self.date_start = date_strat
        self.date_end = date_end
        self.df = pd.DataFrame()
    
    def initialize_df(self):
        pass
    
    def print_df(self, day_of_week = None):

        if day_of_week == None:
            print(self.df)
        else:
            print(self.df[self.df['day_of_week'] == day_of_week])

    def get_df(self):
        return self.df

class Subway_Data(Data):

    def initialize_df(self):

        year_start = int(self.date_start[0:4])  #입력받은 데이터를 년, 월로 쪼갠다. (csv파일을 불러오기 위해)
        month_start = int(self.date_start[4:6])
        year_end = int(self.date_end[0:4])
        month_end = int(self.date_end[4:6])
        months = 12*(year_end-year_start)-month_start+month_end+1 #요청 기간을 달 수로 계산
        df = pd.DataFrame() # 빈 df생성

        for i in range(months):
            if month_start+i > 12: # 1년이 지나면
                year_start += 1 # 연도 + 1
                month_start -= 12 #달 - 12
            if(str(year_start)+str(month_start+i).zfill(2)<='202004'): 
                df_ = pd.read_csv('CARD_SUBWAY_MONTH_'+str(year_start)+str(month_start+i).zfill(2)+'.csv', encoding='CP949', 
                names=['date', 'line', 'station_id', 'station_name', 'num_on', 'num_off', 'date_rgs'] ) #파일 읽어오고 저장
                df_ = df_.drop(0)
                 
                df_ = df_.drop('station_id', axis=1)
                
            elif(str(year_start)+str(month_start+i).zfill(2)<='202005'): #5월 데이터는 역 id를 제공하지 않음
                df_ = pd.read_csv('CARD_SUBWAY_MONTH_'+str(year_start)+str(month_start+i).zfill(2)+'.csv', encoding='CP949', 
                names=['date', 'line', 'station_name', 'num_on', 'num_off', 'date_rgs'] )
                df_ = df_.drop(0)
                
            if i == 0: #선택된 날짜까지만 저장
                df_ = df_[df_['date'] >= self.date_start]
            if i == months-1:
                df_ = df_[df_['date'] <= self.date_end]
                
            df = pd.concat([df, df_], ignore_index=True)  #달로 분리된 df 합침

        df['num_pass'] = df['line']# num_pass column생성
        
        for i in range(len(df)):
            df['num_pass'].loc[i] = int(float(df['num_on'].loc[i])) + int(float(df['num_off'].loc[i])) 
            #float로 바꾸고 int로 바꿔야함
            #승하차객 합쳐서 이용객으로 취급
            
        df['date'] = pd.to_datetime(df['date'], errors='coerce') #datetime 객체로 바꿔주고 
        df['day_of_week'] = df['date'].dt.day_name()# 요일 계산하여 column추가
        
        #필요없는 데이터 제거
        df = df.drop('num_on', axis=1)
        df = df.drop('num_off', axis=1)
        df = df.drop('date_rgs', axis=1)
        
        
        self.df = df

    def get_subway_daily_df(self): #역당 일일 평균 이용객을 구하여 df만드는 함수

        df = self.df
        result_df = pd.DataFrame()
        converted_date_start = dt.datetime.strptime(self.date_start, '%Y%m%d').date() #string날짜를 datetime객체로 바꿔줌
        converted_date_end = dt.datetime.strptime(self.date_end, '%Y%m%d').date()
        days = (converted_date_end-converted_date_start).days
        
        for i in range(days+1):
            
           date = converted_date_start + dt.timedelta(days=i)
           converted_date = date.strftime('%Y%m%d')
           #print(converted_date)
           df_ = df[df['date'] == converted_date]
           df__ = pd.DataFrame(data={'date': [date], 'num_pass': [df_['num_pass'].mean()], 'day_of_week' : [date.weekday()]}) #date.weekday()
           #num_pass 는 모든 역의 일일 평균 이용객 수임
           result_df = pd.concat([result_df, df__], ignore_index=True)
           #if i == 1:
           # print(df_)
           # print(df_['num_pass'].mean())
            
        return result_df

    def get_num_pass_mean(self, day_of_week = None):

        if day_of_week == None:
            return self.df['num_pass'].mean()
        else:
            return self.df[self.df['day_of_week'] == day_of_week]['num_pass'].mean()



class Corona_Data(Data):

    def initialize_df(self):

        date_start = self.date_start[0:4] + '-' + self.date_start[4:6] + '-' + self.date_start[6:8]
        date_end = self.date_end[0:4] + '-' + self.date_end[4:6] + '-' + self.date_end[6:8]

        df = pd.read_csv('wuhan_daily_diff.csv', encoding='CP949', names=['date', 'inspected', 'negative',
        'confirmed', 'recoverd', 'deaths'])

        df = df.drop(0)
        df = df[df['date'] >= date_start]
        df = df[df['date'] <= date_end]

        df['date'] = pd.to_datetime(df['date'], errors='coerce') #datetime type으로 바꿈
        df['day_of_week'] = df['date'].dt.day_name()
        df['confirmed'] = df['confirmed'].astype(float)
        df = df.reset_index(drop=True)

        self.df = df 

#지하철, 코로나 데이터를 하나의 dataframe으로 만들어주는 함수

def concatenate(date_start, date_end, factor=0, drop_holiday=False):
    
    test_subway = Subway_Data(date_start, date_end) 
    test_corona = Corona_Data(date_start, date_end)
    test_subway.initialize_df()
    test_corona.initialize_df()
    test_df = pd.concat([test_subway.get_subway_daily_df().set_index('date'), test_corona.get_df().set_index('date')], axis = 1)
    test_df = test_df.loc[:,~test_df.columns.duplicated()]
    test_df = test_df.drop('inspected', axis=1)
    test_df = test_df.drop('negative', axis=1)
    test_df.index = pd.to_datetime(test_df.index)

    if factor != 0 :
        #가중치 구하기
        test_weekday = Subway_Data('20170101', '20171231') #과거지하철 데이터 생성
        test_weekday.initialize_df()
        weekday_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        num_pass_mean_list = [] #요일별 이용객 수가 저장될 리스트
        weight_list = [] #요일별 가중치가 저장될 리스트
        
        for weekday in weekday_list:
            num_pass_mean_list.append(test_weekday.get_num_pass_mean(day_of_week=weekday)) #Subway_Data의 메소드를 사용하여 요일별 평균을 구함

        mean = sum(num_pass_mean_list)/len(num_pass_mean_list) #요일별 평균 이용객 수의 평균을 구한다.

        for num_pass_mean in num_pass_mean_list:
            weight_list.append((mean/num_pass_mean)**factor) #weight = 요일별 이용객 수 평균의 평균/요일별 평균

        #가중치 곱해주기 
        i = 0
        for weight in weight_list:
            test_df.loc[test_df['day_of_week'] == i, 'num_pass'] = test_df.loc[test_df['day_of_week'] == i, 'num_pass'] * weight
            i += 1

    if drop_holiday == True :
        #주말 이용객 수를 0으로 만듦
        len_df = len(test_df)
        for i in range(len_df):
            if test_df.iloc[i, 1] == 5 : #index의 요일(week_of_day)이 토요일일 때
                if 0 < i < len_df-2 :
                    test_df.iloc[i, 0] = 0 #test_df.iloc[i-1,0] + (1/3)*(test_df.iloc[i+2,0] - test_df.iloc[i-1,0])
                    #금요일과 월요일의 이용객 수 차를 1:2로 내분하는 지점 + 금요일 이용객 수
                elif i == 0 :
                    test_df.iloc[i, 0] = test_df.iloc[i+2, 0]
                elif i >= len_df -2 :
                    test_df.iloc[i, 0] = test_df.iloc[i-1, 0]
            elif test_df.iloc[i, 1] == 6 :
                if 1 < i < len_df -1 :
                    test_df.iloc[i, 0] = 0  #test_df.iloc[i+1,0] - (1/3)*(test_df.iloc[i+1,0] - test_df.iloc[i-2,0]) # 2:1로 내분하는 지점
                elif i <= 1 :
                    test_df.iloc[i, 0] = test_df.iloc[i+1, 0]
                elif i >= len_df -1 :
                    test_df.iloc[i, 0] = test_df.iloc[i-2, 0]
       
        #4.30 부처님오신날
        #5.1 근로자의날
        #5.2 토요일 5.3 일요일
        #5.4 연휴
        #5.5 어린이날
        #이 구간만 추출하면 에러 발생할 수 있음
        if date_end == '20200430' > date_start:
            test_df.loc['2020-04-30', 'num_pass'] = test_df.loc['2020-04-29', 'num_pass']
        elif date_end > '20200504' and '20200430' > date_start :
            test_df.loc['2020-04-30', 'num_pass'] = test_df.loc['2020-04-29', 'num_pass'] + (1/7)*( test_df.loc['2020-05-06', 'num_pass']- test_df.loc['2020-04-29', 'num_pass'])
            test_df.loc['2020-05-01', 'num_pass'] = test_df.loc['2020-04-29', 'num_pass'] + (2/7)*( test_df.loc['2020-05-06', 'num_pass']- test_df.loc['2020-04-29', 'num_pass'])
            test_df.loc['2020-05-04', 'num_pass'] = test_df.loc['2020-04-29', 'num_pass'] + (5/7)*( test_df.loc['2020-05-06', 'num_pass']- test_df.loc['2020-04-29', 'num_pass'])
            test_df.loc['2020-05-05', 'num_pass'] = test_df.loc['2020-04-29', 'num_pass'] + (6/7)*( test_df.loc['2020-05-06', 'num_pass']- test_df.loc['2020-04-29', 'num_pass'])
        
        #4.15 총선   
        if date_end == '20200415' > date_start :
            test_df.loc['2020-04-15', 'num_pass'] = test_df.loc['2020-04-14', 'num_pass'] 
        elif date_end > '20200415' > date_start :
            test_df.loc['2020-04-15', 'num_pass'] = 0.5*(test_df.loc['2020-04-14', 'num_pass'] + test_df.loc['2020-04-16', 'num_pass'])

        for i in range(len_df):
            if test_df.iloc[i, 1] == 5 : #index의 요일(week_of_day)이 토요일일 때
                if 0 < i < len_df-2 :
                    test_df.iloc[i, 0] = test_df.iloc[i-1,0] + (1/3)*(test_df.iloc[i+2,0] - test_df.iloc[i-1,0])
                    #금요일과 월요일의 이용객 수 차를 1:2로 내분하는 지점 + 금요일 이용객 수
                elif i == 0 :
                    test_df.iloc[i, 0] = test_df.iloc[i+2, 0]
                elif i >= len_df -2 :
                    test_df.iloc[i, 0] = test_df.iloc[i-1, 0]
            elif test_df.iloc[i, 1] == 6 :
                if 1 < i < len_df -1 :
                    test_df.iloc[i, 0] = test_df.iloc[i+1,0] - (1/3)*(test_df.iloc[i+1,0] - test_df.iloc[i-2,0]) # 2:1로 내분하는 지점
                elif i <= 1 :
                    test_df.iloc[i, 0] = test_df.iloc[i+1, 0]
                elif i >= len_df -1 :
                    test_df.iloc[i, 0] = test_df.iloc[i-2, 0]


    test_df = test_df.drop('day_of_week', axis=1)
    
    return test_df


"""
소스 출처 : https://frhyme.github.io/machine-learning/regression_evaluation_score/
"""
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
    
def PrintRegScore(y_true, y_pred):
    print('explained_variance_score: {}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_errors: {}'.format(mean_squared_error(y_true, y_pred)))
    print('r2_score: {}'.format(r2_score(y_true, y_pred)))


"""
소스 출처 https://frhyme.github.io/machine-learning/regression_evaluation_score/
"""


def main():
  
    return 0

if __name__ == "__main__":
    main()

   