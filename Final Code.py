#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
import re 
import tweepy
import seaborn as sb
import warnings
import nltk
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, explained_variance_score, mean_absolute_error
from scipy.cluster.vq import kmeans, vq, whiten
from scipy import stats
from sklearn import preprocessing, tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
warnings.filterwarnings('ignore')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import plotly.figure_factory as ff
import plotly as py
from sklearn.tree import DecisionTreeClassifier


# In[2]:


order_item = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_order_items_dataset.csv')
order = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_orders_dataset.csv')
product = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_products_dataset.csv')
customer = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_customers_dataset.csv')
geolocation = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_geolocation_dataset.csv')
sellers = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_sellers_dataset.csv')
product_category = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/product_category_name_translation.csv')
order_reviews = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_order_reviews_dataset.csv')
order_payment = pd.read_csv('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/olist_order_payments_dataset.csv')
sentimental_analysis = pd.read_excel('/Users/dibshrestha/Documents/Summer/Python/brazilian-ecommerce/Sentimental_Analysis.xlsx','Sheet1')



# In[3]:


order_reviews.head()


# In[4]:


date_format = '%Y-%m-%d %H:%M:%S'


# #           Which Product Category does not get delivered on time?

# In[5]:


#Merging tables necessary 
product_category_delivery = pd.merge(order, order_item, on='order_id',how='inner')
product_category_delivery = pd.merge(product_category_delivery, product, on='product_id',how='inner')
product_category_delivery = pd.merge(product_category_delivery, product_category, on='product_category_name',how='inner')


# In[6]:


#Removing Null values in order_estimated_delivery_date and order_delivered_customer_date
product_category_delivery = product_category_delivery[product_category_delivery.order_estimated_delivery_date.notnull()]
product_category_delivery = product_category_delivery[product_category_delivery.order_delivered_customer_date.notnull()]


# In[7]:


#Calculating Delayed Days (order_delivered_customer_date - order_estimated_delivery_date)
product_category_delivery['delayed_days'] = product_category_delivery.apply(lambda row: (datetime.strptime(str(row.order_delivered_customer_date), date_format) - datetime.strptime(str(row.order_estimated_delivery_date), date_format)), axis=1)
product_category_delivery['delayed_days'] = product_category_delivery.apply(lambda row: row.delayed_days.days, axis=1)


# In[8]:


#Removing Unnecessary Columns
product_category_delivery.drop(['product_category_name', 'seller_id', 'shipping_limit_date', 'price', 'customer_id', 
                                'order_status', 'order_purchase_timestamp', 'order_approved_at', 
                                'order_delivered_carrier_date', 'order_delivered_customer_date', 
                                'order_estimated_delivery_date', 'freight_value', 'product_name_lenght', 
                                'product_description_lenght', 'product_photos_qty', 'product_weight_g', 
                                'product_length_cm', 'product_height_cm', 'product_width_cm'], axis=1, inplace=True)


# In[9]:


product_category_delivery['order_id'].value_counts().head()


# In[10]:


#Since some orders have multiple product ids those would have the same delivery date and it wouldn't make sense to keep them
product_category_delivery.loc[product_category_delivery['order_id'] == '30bdf3d824d824610a49887486debcaf'].head(7)


# In[11]:


#Removing orders which has more than 1 items
product_category_delivery = product_category_delivery[~product_category_delivery.order_id.isin(product_category_delivery[product_category_delivery.order_item_id > 1]['order_id'])]


# In[12]:


product_category_delivery['order_id'].value_counts().head()


# In[13]:


product_category_delivery.info()
product_category_delivery.drop(['order_id', 'order_item_id', 'product_id'], axis=1, inplace=True)


# In[14]:


product_category_delivery.head()


# In[15]:


#Removing Orders that are delivered on time
product_category_delivery = product_category_delivery.query('delayed_days > 0')


# In[16]:


#Changing mean and row count for each Product Category
product_category_delivery_mean = product_category_delivery.groupby(['product_category_name_english']).mean().round(2)
product_category_delivery_count = product_category_delivery.groupby(['product_category_name_english']).count()


# In[17]:


#Merging count and Mean Delayed Delivery Date
product_category_delivery = pd.merge(product_category_delivery_mean, product_category_delivery_count, on='product_category_name_english',how='inner')
product_category_delivery = product_category_delivery.rename(columns={"delayed_days_x": "mean_Delivery_Delayed_Days", "delayed_days_y": "order_count"})


# In[18]:


product_category_delivery.sort_values(by=('mean_Delivery_Delayed_Days'),ascending=False).head()


# In[19]:


#Graph For the Question 


# # Which customer area does not get delivered on time?

# In[20]:


customer_delivery = pd.merge(order, order_item, on='order_id',how='inner')
customer_delivery = pd.merge(customer_delivery, customer, on='customer_id', how='inner')

customer_delivery = customer_delivery[customer_delivery.order_estimated_delivery_date.notnull()]
customer_delivery = customer_delivery[customer_delivery.order_delivered_customer_date.notnull()]


customer_delivery['delayed_days'] = customer_delivery.apply(lambda row: (datetime.strptime(str(row.order_delivered_customer_date), date_format) - datetime.strptime(str(row.order_estimated_delivery_date), date_format)), axis=1)
customer_delivery['delayed_days'] = customer_delivery.apply(lambda row: row.delayed_days.days, axis=1)

customer_delivery.drop(['price', 'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                        'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                        'order_estimated_delivery_date', 'order_item_id', 'product_id', 'seller_id', 
                        'shipping_limit_date', 'freight_value', 'customer_unique_id', 'customer_zip_code_prefix', 
                        'customer_state'], axis=1, inplace=True)

customer_delivery_days = customer_delivery.groupby(['customer_city']).mean().round(2)
customer_delivery_count = customer_delivery.groupby(['customer_city']).count()


customer_delivery = pd.merge(customer_delivery_days, customer_delivery_count, on='customer_city',how='inner')
customer_delivery = customer_delivery.rename(columns={"delayed_days_x": "mean_Delivery_Delayed_Days", "delayed_days_y": "order_count"})


# In[21]:


customer_delivery.sort_values(by=('mean_Delivery_Delayed_Days'),ascending=False).head()


# In[22]:


customer_delivery.sort_values(by=('order_count'),ascending=False).head()


# In[23]:


#Could add some graph here


# # Which Area had most Order From?

# In[24]:


#Merging Necessary Tables
customer_area = pd.merge(order, order_item, on='order_id',how='inner')
customer_area = pd.merge(customer_area, customer, on='customer_id', how='inner')


# In[25]:


#Drop Columns
customer_area.drop(['price', 'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                        'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 
                        'order_estimated_delivery_date', 'order_item_id', 'product_id', 'seller_id', 
                        'shipping_limit_date', 'freight_value', 
                        'customer_state', 'customer_zip_code_prefix', 'customer_unique_id'], axis=1, inplace=True)


# In[26]:


#Group by and count of Customer Cites
customer_order_count = customer_area.groupby(['customer_city'])['customer_city'].count()


# In[27]:


#Customer Cities with highest Orders
customer_order_count.sort_values(ascending=False).head(10)


# # Which Area had most items shipped from?

# In[28]:


#Merging Necessary Tables
seller_area = pd.merge(order, order_item, on='order_id',how='inner')
seller_area = pd.merge(seller_area, sellers, on='seller_id', how='inner')


# In[29]:


#Drop Columns
seller_area.drop(['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 
                  'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 
                  'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value', 
                  'seller_zip_code_prefix', 'seller_state' ], axis=1, inplace=True)


# In[30]:


#Group by and count of Seller Cites
seller_area_count = seller_area.groupby(['seller_city'])['seller_city'].count()


# In[31]:


#Seller Cities with highest Orders
seller_area_count.sort_values(ascending=False).head(10)


# In[32]:


#Graph


# # Payment Types

# In[33]:


ax = sb.violinplot(x="payment_type", y="payment_value", data=order_payment, inner=None)
ax.set_ylim([-150,900])


# # Authenticity Of Review Score

# In[34]:


# Merging the order reviews and order files 
authenticity_Review = pd.merge(order,order_reviews)
authenticity_Review.head()


# In[35]:


#Checking for Unique values in "Order Status"
authenticity_Review.order_status.unique()


# In[36]:


#Checking for Unique values in "Review Score"
authenticity_Review.review_score.unique()


# In[37]:


#Summarized Order Status Vs Review Score
authenticity_Review.pivot_table(index='order_status')


# In[38]:


#Number of Orders Per Status

Order_status_count = authenticity_Review.groupby(['order_status'], sort=False)['review_score'].count().reset_index()
Order_status_count = Order_status_count.sort_values(by='review_score', ascending=False)
print(Order_status_count)


# In[39]:


#Filtering for "Canceled" orders

NP=authenticity_Review[authenticity_Review.order_status == 'canceled']
Fraud=NP[NP.review_score >= 4]
Fraud= Fraud[['order_status','review_score']]
Cancelled_count = Fraud.groupby(['order_status'], sort=False)['review_score'].count().reset_index()
print(Cancelled_count)


# In[40]:


#Filtering for "Unavailable" orders

MP=authenticity_Review[authenticity_Review.order_status == 'unavailable']
Fraud_1=MP[MP.review_score >= 4]
Fraud_1
Fraud_1= Fraud_1[['order_status','review_score']]
Unavailable_count = Fraud_1.groupby(['order_status'], sort=False)['review_score'].count().reset_index()
print(Unavailable_count)


# # Sentiment Analysis On Review Messages

# In[41]:


#Data Cleaning
sentiment_Analysis = pd.DataFrame(sentimental_analysis.drop(['Unnamed: 3'],axis=1,inplace=False))


#Review Score counts
sentiment_Analysis['review_score'].value_counts().plot(kind='bar')


# In[42]:


#Cleaning Special symbols and converting the review message to lower case
sentiment_Analysis['review_message'] = sentiment_Analysis['review_message'].astype(str)
sentiment_Analysis['review_message'] = sentiment_Analysis['review_message'].apply(lambda x: " ".join(x.lower() for x in x.split()))
sentiment_Analysis['review_message'] = sentiment_Analysis['review_message'].str.replace('[^\w\s]','')


# In[43]:


#Eliminating Stopwords from review message
nltk.download('stopwords')
stopwords = stopwords.words('english')
sentiment_Analysis['review_message'] = sentiment_Analysis['review_message'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))


# In[44]:


#Word Stemming using PortStemmer()
st = PorterStemmer()
sentiment_Analysis['review_message'] = sentiment_Analysis['review_message'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[45]:


#Sentiment Analysis on review message

def senti(x):
    return TextBlob(x).sentiment  
 
sentiment_Analysis['senti_score'] = sentiment_Analysis['review_message'].apply(senti)
 
sentiment_Analysis.senti_score.head()


# In[46]:


#Importing Stopwords for Word Cloud
stopwords = set(STOPWORDS)


# In[47]:


#Generating Word Cloud
def show_wordcloud(col, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=500,
        max_font_size=40, 
        scale=3,
        random_state=1
    ).generate(str(col))

    fig = plt.figure(1, figsize=(14, 14))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


if __name__ == '__main__':

    show_wordcloud(sentiment_Analysis['review_message'])


# # What are review score based on?

# In[48]:


review_score = pd.merge(order_item, order, on='order_id', how='inner')
review_score = pd.merge(review_score, order_reviews, on='order_id', how='inner')


# In[49]:


review_score.drop(['order_item_id', 'review_id', 'review_comment_title', 'review_comment_message', 'review_answer_timestamp', 
                   'review_creation_date', 'shipping_limit_date', 'seller_id', 'order_delivered_carrier_date'
                 ], axis=1, inplace=True)

review_score = review_score[review_score.order_status == 'delivered']


# In[50]:


review_score['order_purchase_date']= review_score['order_purchase_timestamp'].astype("datetime64")
review_score['approval_date'] = review_score['order_approved_at'].astype('datetime64')
review_score['delivery_date'] = review_score['order_delivered_customer_date'].astype('datetime64')
review_score['estimated_delivery'] = review_score['order_estimated_delivery_date'].astype('datetime64')


# In[51]:


review_score.drop(['order_purchase_timestamp', 'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date'
                 ], axis=1, inplace=True)


# In[52]:


review_score['Approval_time_hours'] = (review_score['approval_date'] - review_score['order_purchase_date'])
review_score['Delivery_time_days'] = (review_score['delivery_date'] - review_score['order_purchase_date'])
review_score['Ontime_Delivery'] = (review_score['delivery_date']<= review_score['estimated_delivery']).astype(int)

review_score['Approval_time_hours'] = review_score['Approval_time_hours'] / np.timedelta64(1, 'h')
review_score['Delivery_time_days'] = review_score.apply(lambda row: row.Delivery_time_days.days, axis=1)
review_score.head()


# In[53]:


bins_review_score = np.linspace(min(review_score["review_score"]), max(review_score["review_score"]), 6)
review_score_groups = ["one","Two", "Three", "Four", "Five"]
review_score['review_binned'] = pd.cut(review_score['review_score'], bins_review_score,labels = review_score_groups, include_lowest = True)


# In[54]:


bins_price = np.linspace(min(review_score["price"]), max(review_score["price"]), 6)
price_groups = ["very low","low", "medium", "high", "very high"]
review_score['price_binned'] = pd.cut(review_score['price'], bins_price,labels = price_groups, include_lowest = True)


# In[55]:


bins_ontime = np.linspace(0, 1, 3)
ontime_groups = ["No","Yes"]
review_score['ontime_binned'] = pd.cut(review_score['Ontime_Delivery'], bins_ontime,labels = ontime_groups, include_lowest = True)


# In[56]:


review_score['price_scaled']  = review_score['price']/review_score['price'].mean()


# In[57]:


x_anova = review_score[['price_binned', 'review_score']]
grouped_anova = x_anova.groupby(["price_binned"])


# In[58]:


anova_results_1 = stats.f_oneway(grouped_anova.get_group("low")['review_score'], grouped_anova.get_group("medium")['review_score'])
print(anova_results_1)


# In[59]:


anova_results_1 = stats.f_oneway(grouped_anova.get_group("low")['review_score'], grouped_anova.get_group("high")['review_score'])
print(anova_results_1)


# In[60]:


anova_results_1 = stats.f_oneway(grouped_anova.get_group("low")['review_score'], grouped_anova.get_group("very high")['review_score'])
print(anova_results_1)


# In[61]:


anova_results_1 = stats.f_oneway(grouped_anova.get_group("very low")['review_score'], grouped_anova.get_group("low")['review_score'])
print(anova_results_1)


# In[62]:


sb.pairplot(review_score)


# In[63]:


review_score = review_score.drop(['order_id','customer_id', 'order_status','order_purchase_date','approval_date', 'delivery_date', 'estimated_delivery', 'product_id','review_binned','price_binned','ontime_binned','price'], axis = 1)
review_score = review_score.sort_values('Approval_time_hours')
review_score = review_score.dropna()


# In[64]:


corrs = review_score.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()


# In[65]:


sb.regplot(x = review_score['review_score'], y = review_score['Delivery_time_days'])


# In[66]:


sb.regplot(x = review_score['review_score'], y = review_score['Ontime_Delivery'])


# # Decision Tree to Analysis Rating Based on Price and Freight Value

# In[67]:


decision_tree = pd.merge(order_item, order_reviews, on='order_id',how='inner')
decision_tree = decision_tree[~decision_tree.order_id.isin(decision_tree[decision_tree.order_item_id > 1]['order_id'])]

decision_tree.drop(['order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 
                    'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date', 
                    'review_answer_timestamp' ], axis=1, inplace=True)

decision_tree.head()


# In[68]:


decision_tree['Rating'] = np.where(decision_tree['review_score']>3, 'Good', 'Bad')



target=decision_tree['Rating']


# In[69]:


decision_tree.drop(['review_score', 'Rating' ], axis=1, inplace=True)


# In[70]:


model= tree.DecisionTreeClassifier()
model.fit(decision_tree,target)


# In[71]:


model.score(decision_tree,target)


# In[72]:


X_train, X_test, y_train, y_test= train_test_split(decision_tree,target,test_size=0.20,random_state=1)


# In[73]:


model_1= tree.DecisionTreeClassifier(max_depth=3)
model_1.fit(X_train, y_train)
y_pred= model_1.predict(X_test)


# In[74]:


print("Accuracy:", accuracy_score(y_test, y_pred))


# In[75]:


dot_data = StringIO()
export_graphviz(model_1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# # Predicting Freight Value

# In[76]:


#Order Ids have multiple items
order_item['order_id'].value_counts().head(10)


# In[77]:


#Different Product will have same freight_value if they are ordered together
order_item.loc[order_item['order_id'] == '8272b63d03f5f79c56e9e4120aec44ef'][['product_id', 'freight_value']].head(2)


# In[78]:


#Removing the orders which has more than 1 item
freight_value = order_item[~order_item.order_id.isin(order_item[order_item.order_item_id > 1]['order_id'])]
freight_value.info()


# In[79]:


#Count of Product Sold
freight_value['product_id'].value_counts().head(10)


# In[80]:


#Checking Freight Value for a same product
freight_value.loc[freight_value['product_id'] == '99a4788cb24856965c36a24e339b6058'][['product_id', 'freight_value']].head()


# In[81]:


#Merging necessary tables
geolocation_freight=geolocation.groupby('geolocation_zip_code_prefix').aggregate('mean')
freight_value = pd.merge(freight_value, order, on='order_id', how='inner')
freight_value = pd.merge(freight_value, product, on='product_id', how='inner')
freight_value = pd.merge(freight_value, customer, on='customer_id', how='inner')
freight_value = pd.merge(freight_value, sellers, on='seller_id', how='inner')
freight_value = pd.merge(freight_value, geolocation_freight, left_on='customer_zip_code_prefix', right_index=True, how='inner')
freight_value = pd.merge(freight_value, geolocation_freight, left_on='seller_zip_code_prefix', right_index=True, how='inner')


# In[82]:


#Removing Unnecessary Columns
freight_value.drop(['order_item_id', 'shipping_limit_date', 'price', 'order_status', 'order_purchase_timestamp', 
                 'order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
                 'order_estimated_delivery_date', 'product_category_name', 'product_name_lenght', 
                 'product_description_lenght','product_photos_qty', 'customer_unique_id', 
                 'customer_city', 'customer_state', 'seller_city', 'seller_state', 'seller_id', 'customer_id', 
                 ], axis=1, inplace=True)


freight_value.head()


# In[83]:


#Calculating Distance and Volume
freight_value['distance'] = freight_value.apply(lambda row: ((row.geolocation_lat_x - row.geolocation_lat_y)**2+(row.geolocation_lng_x - row.geolocation_lng_y)**2)**0.5, axis=1)
freight_value['product_volume'] = freight_value.apply(lambda row: (row.product_length_cm * row.product_height_cm * row.product_width_cm), axis=1)
freight_value['distance'] = freight_value['distance'].round(3)



# In[84]:


#Comparing Freight_Value and Distance for the same Product
single_product = freight_value.loc[freight_value['product_id'] == '99a4788cb24856965c36a24e339b6058'][['product_id', 'freight_value', 'distance']].sort_values(by ='freight_value')
single_product.head()


# In[85]:


#Comparing Freight_Value and Distance for the same Product
sb.regplot(x = single_product['distance'], y = single_product['freight_value'])
plt.title('Distance VS Freight Value for a single product')
plt.show()


# In[86]:


freight_value['distance'].value_counts().head(10)


# In[87]:


#Taking two freight value
single_freight_value = freight_value.loc[(freight_value['distance'] == 0.166) | (freight_value['distance'] == 0.165)][['product_weight_g', 'product_volume', 'freight_value']].sort_values(by ='product_volume')


# In[88]:


single_freight_value.head()


# In[89]:


#Product Weight VS Freight Value for a single Freight Value
sb.regplot(x = single_freight_value['product_weight_g'], y = single_freight_value['freight_value'])
plt.title('Product Weight VS Freight Value for a Freight Value')
plt.show()


# In[90]:


#Product Weight VS Freight Value for a single Freight Value
plt.title('Product Volume VS Freight Value for a Freight Value')
sb.regplot(x = single_freight_value['product_volume'], y = single_freight_value['freight_value'])
plt.show()


# In[91]:


#Removing Null and free shipping
freight_value = freight_value.dropna(subset=['product_volume', 'product_weight_g', 'geolocation_lat_x', 'geolocation_lng_x', 'geolocation_lat_y', 'geolocation_lng_y'])
freight_value = freight_value[freight_value.freight_value != 0]


# In[92]:


#Separating GeoLocation for Customer and Seller
cust_table = freight_value.filter(['geolocation_lat_x', 'geolocation_lng_x'])
cust_table = cust_table.rename(columns={'geolocation_lat_x': 'latitude', 'geolocation_lng_x': 'longitude'})
prod_table = freight_value.filter(['geolocation_lat_y', 'geolocation_lng_y'])
prod_table = prod_table.rename(columns={'geolocation_lat_y': 'latitude', 'geolocation_lng_y': 'longitude'})


# In[93]:


#K-Means
cust_table = cust_table.append(prod_table)
sci_table = whiten(cust_table)
centroids,_ = kmeans(cust_table, 20)
clx,_ = vq(cust_table, centroids)


# In[94]:


freight_value['customer_area'] = clx[:freight_value.shape[0]]
freight_value['seller_area'] = clx[freight_value.shape[0]:]


# In[95]:


freight_value.head()


# In[96]:


#Customer Area Plot
sb.lmplot('geolocation_lng_x', 'geolocation_lat_x',
           data=freight_value,
           fit_reg=False,
           hue='customer_area', 
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Customer Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[97]:


#Seller Area Plot
sb.lmplot('geolocation_lng_y', 'geolocation_lat_y',
           data=freight_value,
           fit_reg=False,
           hue='seller_area', 
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Seller Area')
plt.xlabel('Longitude')
plt.ylabel('Latitute')
plt.show()


# In[98]:


freight_value.drop(['order_id', 'product_id', 'product_length_cm', 'product_height_cm', 'product_width_cm', 
                    'customer_zip_code_prefix', 'seller_zip_code_prefix',  'geolocation_lat_x', 'geolocation_lng_x', 
                    'geolocation_lat_y', 'geolocation_lng_y'], axis=1, inplace=True)

freight_value.info()


# In[99]:


#Removing Outliers
freight_value = freight_value[(np.abs(stats.zscore(freight_value)) < 3).all(axis=1)]


# In[100]:


#Dividing Freight values in Groups
bins = (0, 10, 20, 30, 60, 100, 450)
group_names = ['0 - 10', '10 - 20', '20 - 30', '30 - 60', '60 - 100', 'More than 100']
freight_value['freight_group'] = pd.cut(freight_value['freight_value'], bins=bins, labels=group_names)


# In[101]:


X = freight_value.filter(['product_volume', 'product_weight_g', 'customer_area', 'seller_area'], axis=1).astype('float64')
y = freight_value['freight_group']


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[103]:


#Scaling the Input
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[104]:


#RandomForestClassifier Model and Accuracy Score
rfc = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("Accuracy Score = ", accuracy_score(y_test, pred_rfc).round(3) )


# In[105]:


Xnew = [[3000, 400, 2, 5]]
Xnew = sc.transform(Xnew)
print("rfc.predict(Xnew) == ", rfc.predict(Xnew))


# In[106]:


#Regression Model
X_regression = freight_value.filter(['product_volume', 'product_weight_g', 'distance'], axis=1)
y_regression = freight_value['freight_value']


# In[107]:


#Preparing Regression Model
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=40)


# In[108]:


#Scaling the Input
sc_regression = StandardScaler()
X_train_regression = sc_regression.fit_transform(X_train_regression)
X_test_regression = sc_regression.transform(X_test_regression)


# In[109]:


y_train_regression= y_train_regression.values.reshape(-1, 1)


# In[110]:


#Linear Regression Model
regression_model = LinearRegression().fit(X_train_regression, y_train_regression)
pred_y = regression_model.predict(X_test_regression)


# In[115]:


#Mean Absolute Error
mean_absolute_error(y_test_regression, pred_y)


# In[116]:


#New Prediction
Xnew = [[3000, 400, 2]]
Xnew = sc_regression.transform(Xnew)
print("model.predict(Xnew) == ", regression_model.predict(Xnew))


# In[ ]:




