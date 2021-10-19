import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
from model1 import model1


st.set_page_config(layout="wide")
def main():
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)

    with col3:
        st.write("")
    st.sidebar.markdown('<p class="big-font">Toyota Forklift Maintainance</p>', unsafe_allow_html=True)
    st.sidebar.write("")
    st.sidebar.write("")


    uploaded_files = st.sidebar.file_uploader("Upload Forklift Data", type=['csv'], accept_multiple_files=False)

    if uploaded_files:
        options=st.sidebar.selectbox('Please Select',['Choose from dropdown','PowerBI','Preprocessing & Predictions','Correlation Matrix'])
        if options=='PowerBI':
            components.html('<iframe width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=1804aa2d-9c6e-450e-8aae-1359783991bf&autoAuth=true&ctid=0f8a5db0-6b60-4ca8-9fc9-8b2ae1cb809e&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9" frameborder="0" allowFullScreen="true"></iframe>', width=1000, height=1000)

        elif options == 'Preprocessing & Predictions':
            pred, train, df = model1(uploaded_files)
            train1 = train.to_numpy().tolist()
            pred1 = pred.tolist()
            train_vehID = [i[0] for i in train1]
            s1 = pd.Series(train_vehID, name='VehicleID')
            s2 = pd.Series(pred1, name='Predicted_Service')
            df_new = pd.concat([s1, s2], axis=1)
            # print(df)

            st.info("Dataset")
            df1 = df.sort_values(by=['veh_id'], ascending=False)
            df_final = df1.head()
            st.dataframe(df_final)

            st.info("Dataset Summary")
            df_desc = df1.describe()
            print(df_desc)
            st.dataframe(df_desc)

            st.info("Predicted Response")
            st.write(df_new.T.to_html(escape=False), unsafe_allow_html=True)


        elif options == 'Correlation Matrix':
            components.html('<iframe width="1040" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=cf478bb2-5a57-49ec-95a3-19e24cb55db2&autoAuth=true&ctid=0f8a5db0-6b60-4ca8-9fc9-8b2ae1cb809e&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9" frameborder="0" allowFullScreen="true">',width=1000, height=1000)


        else:
            pass



            # st.subheader("User Input Parameters")
            # id=st.text_input("Enter Vehicle ID")
            # start_date=st.date_input("Enter start Date")
            # status=st.selectbox("Status",[0,1])
            # logged_status= 1
            # Total_dist=st.text_input("Total Distance Covered")
            # Avg_run_time=st.text_input("Avg Run Time")
            # fuel=st.selectbox("Fuel level",[0,1,2])
            # lubricant_change=st.date_input("Enter the Lubricant last change date")
            # lubricant_due=st.date_input("Enter lubricant due date")
            # tyre=st.slider("Pressure Level in PSI",0.0,150.0,10.0)
            # battery_change=st.date_input("Enter Battery Change date")
            # battery_due=st.date_input("Enter Battery Change Due Date")
            # brakes=st.selectbox("Condition of brakes",[0,1])
            # temp=st.selectbox("Temperature ",[0,1,2])
            # vibration = st.selectbox("Vibration", [0, 1, 2])
            # coolant = st.selectbox("Coolant Level", [0, 1, 2])
            # Total_work_received=st.text_input("Total Work Order Recieved")
            # Total_work_Completed = st.text_input("Total Work Order Completed")
            # data={'veh_id':id,'Start_Date':start_date,'Status':status
            #      ,'Logged status':logged_status,'Total_Dist_Covered(km)':Total_dist
            #      ,'Avg_Run_Time':Avg_run_time,'Fuel Level':fuel,'Lubricants_last_changed':lubricant_change
            #      ,'Lubricant_due_date':lubricant_due,'Tyre_pressure(PSI)':tyre,'Battery_change_Date':battery_change
            #      ,'Battery_change_due_date':battery_due,'Brakes':brakes,'Temperature':temp,'Vibration':vibration,'Coolant_level':coolant
            #      ,'Total_Work_order_Received':Total_work_received,'Total_Work_Order_Completed':Total_work_Completed}

        # print(data[-1])
            # features=pd.DataFrame(data)
            # print(features)
            # # return features
            #
            # # st.subheader("User Input Parameters")
            # # df1= user_value()
            #
            # st.write(features)
            # y = features.iloc[:, -1:]
            # x = features.iloc[:, 0:-1]
            # print(x,y)
            #
            # # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
            #
            #
            # load_model=pickle.load(open('finalized_model.pkl','rb'))
            # prediction=load_model.predict(features)
            # prediction_proba=load_model.predict_proba(features)
            # st.subheader("Prediction Probability")
            # st.write(prediction_proba)



if __name__ == '__main__':
    main()