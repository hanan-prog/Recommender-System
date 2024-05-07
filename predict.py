import time, json, pyspark


def predict_rating(pair, user_b_rating, in_model, n_weights, avg_business_rating, avg_user_rating, avg_ratings):

    user_id, business_id = pair[0], pair[1]
    # bxi = overall average + bx + bi
    base_est = avg_ratings + avg_user_rating[user_id] + avg_business_rating[business_id]
    
    n_similar = []
    # find business rated by user and in model
    for business in user_b_rating[user_id]:
        sim_business = (business, business_id)
        if sim_business in in_model:
            rating = user_b_rating[user_id][business] - base_est
            n_similar.append((business, in_model[sim_business], rating))

    # sort by similarity and take top n
    n_similar = sorted(n_similar, key=lambda x: x[1], reverse=True)
    n_similar = n_similar[:n_weights]       # n_similar = [(business, sim, rating), ...] 
    
    numerator = 0
    denominator = 0
    for (business, sim, rating) in n_similar:
        item_correction = avg_business_rating[business] 
        numerator += (rating - item_correction) * sim
        # numerator += rating * sim
        denominator += sim

    
    if denominator == 0:
        # base_est = round(base_est, 2)
        if base_est > 5:
            base_est = 5.0
        elif base_est < 1:
            base_est = 1.0
        
    
        return (user_id, business_id, base_est)
    
    pred = base_est + (numerator / denominator)
    if pred > 5:
        pred = 5.0
    elif pred < 1:
        pred = 1.0 
    return (user_id, business_id, pred)



def main(train_file, test_file, model_file, output_file, n_weights, sc):
    in_train = sc.textFile(train_file).map(lambda x: json.loads(x))\
        .persist()
    
    in_model = sc.textFile(model_file).map(lambda x: json.loads(x))\
        .map(lambda x: ((x['b1'], x['b2']), x['sim']))\
        .collectAsMap()
    
    in_test = sc.textFile(test_file).map(lambda x: json.loads(x))\
        .map(lambda x: (x['user_id'], x['business_id'])).persist()
    
    user_b_rating = in_train.map(lambda x: (x['user_id'], (x['business_id'], x['stars'])))\
        .groupByKey().mapValues(dict).collectAsMap()
    
    # overall average rating
    ratings = in_train.map(lambda x: x['stars']).collect()
    avg_ratings = sum(ratings) / len(ratings)

    # bx = rating deviation of user x from the overall average
    avg_user_rating = in_train.map(lambda x: (x['user_id'], x['stars']))\
        .groupByKey().mapValues(lambda x: ((sum(x) / len(x)) - avg_ratings))\
        .collectAsMap()
    
    
    
    avg_business_rating = in_train.map(lambda x: (x['business_id'], x['stars']))\
        .groupByKey().mapValues(lambda x: ((sum(x) / len(x)) - avg_ratings))\
        .collectAsMap()
    

    predictions = in_test\
        .map(lambda x: (predict_rating(x, user_b_rating, in_model, n_weights, avg_business_rating, avg_user_rating, avg_ratings)))\
        .persist()



    predictions = predictions.map(lambda x: {'user_id': x[0], 'business_id': x[1], 'stars': x[2]})\
        .collect()
    
    with open(output_file, 'w') as outfile:
        for line in predictions:
            outfile.write(json.dumps(line) + '\n')
    

if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task2') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")


    train_file = './data/train_review.json'
    test_file = './data/test_review.json'
    model_file = 'rating.model'
    output_file = 'out.json'
    n = 30
    main(train_file, test_file, model_file, output_file, n, sc)
    sc.stop()