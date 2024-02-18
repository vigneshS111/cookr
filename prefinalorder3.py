import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from operator import itemgetter

# Define training data based on rules
rules_data = {
    'Order_Source': ['Same Kitchen', 'Different Kitchens', 'Same Kitchen', 'Different Kitchens', 'Different Kitchens', 'Same Kitchen', 'Different Kitchens', 'Same Kitchen'],
    'Order_Destination': ['Same Customer', 'Same Customer', 'Different Customers', 'Same Customer', 'Same Customer', 'Same Customer', 'Different Customers', 'Different Customers'],
    'Distance_Between_Kitchens': [0, 1, 0, 1, 1, 0, 1, 0],
    'Ready_Time_Difference': [10, 10, 10, 10, 10, 10, 10, 10],
    'Pickup_Strategy': ['Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider', 'Same Rider'],
    'Rule': [1, 2, 3, 4, 5, 6, 7, 8]
}

# Create a DataFrame from the rules_data
df_rules = pd.DataFrame(rules_data)

# Display the dataset
print(df_rules)

# Define features (excluding 'Rule') and target variable ('Rule')
features = ['Order_Source', 'Order_Destination', 'Distance_Between_Kitchens', 'Ready_Time_Difference', 'Pickup_Strategy']
target = 'Rule'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_rules[features], df_rules[target], test_size=0.2, random_state=42)

# Define a transformer to apply one-hot encoding to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Order_Source', 'Order_Destination', 'Pickup_Strategy'])
    ],
    remainder='passthrough'
)

# Create a pipeline with the transformer and logistic regression model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model to the training data
model.fit(X_train, y_train)

orders = [
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 0, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 10},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 5},
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 15},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 1, 'Ready_Time_Difference': 10, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 8},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 2, 'Ready_Time_Difference': 15, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 12},
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 3, 'Ready_Time_Difference': 12, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 4},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 5, 'Ready_Time_Difference': 20, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 18},
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Different Customers', 'Distance_Between_Kitchens': 6, 'Ready_Time_Difference': 25, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 20},
    {'Order_Source': 'Different Kitchens', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 7, 'Ready_Time_Difference': 30, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 2},
    {'Order_Source': 'Same Kitchen', 'Order_Destination': 'Same Customer', 'Distance_Between_Kitchens': 8, 'Ready_Time_Difference': 8, 'Pickup_Strategy': 'Same Rider', 'Distance_Between_Customer_And_Kitchen': 7},
]

# Create an order batch data structure based on rules satisfying
order_batches = []

# Simulate the assignment of orders to batches based on the trained model
for order_id, order in enumerate(orders, start=1):
    order_data = pd.DataFrame(order, index=[0])
    assigned_rule = int(model.predict(order_data)[0])

    # Use the assigned rule to determine the batch and rider
    batch = {
        'Order_Destination': order['Order_Destination'],
        'assigned_rule': assigned_rule,
        'rider': None,
        'batch_id': None,
        'order_id': order_id,
        'Distance_Between_Kitchens': order['Distance_Between_Kitchens'],
        'Distance_Between_Customer_And_Kitchen': order['Distance_Between_Customer_And_Kitchen']
    }

    order_batches.append(batch)

# Group orders into batches based on assigned_rule
grouped_orders = {}
for order in order_batches:
    rule = order['assigned_rule']
    grouped_orders.setdefault(rule, []).append(order)

# Assign batch IDs and determine order priority within each batch
for rule, batch_orders in grouped_orders.items():
    # Sort orders within each batch based on priority factors
    sorted_orders = sorted(batch_orders, key=itemgetter(
        'Distance_Between_Customer_And_Kitchen',
        'Distance_Between_Kitchens',
        'Order_Destination',
        'assigned_rule',
        'order_id'
    ))

    # Print the order batches with batch IDs and priority within each batch
    print(f"\nBatch ID: {rule}")
    for order in sorted_orders:
        print(f"Order Destination: {order['Order_Destination']}, Assigned Rule: {order['assigned_rule']}, Distance (Customer-Kitchen): {order['Distance_Between_Customer_And_Kitchen']}, Distance (Kitchen): {order['Distance_Between_Kitchens']}, Order ID: {order['order_id']}")
