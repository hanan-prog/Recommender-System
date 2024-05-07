import time, json, pyspark, random, itertools

def jaccard_sim(org_matrix, cand_pair):
    business1 = set(org_matrix[cand_pair[0]])
    business2 = set(org_matrix[cand_pair[1]])
    intersection = len(business1.intersection(business2))
    union = len(business1.union(business2))
    if union == 0:
        return 0
    
    return (cand_pair[0], cand_pair[1], intersection / union)

def create_sig(sc, input_file):
    in_file = sc.textFile(input_file).map(lambda x: json.loads(x))\
        .persist()
    
    users = in_file.map(lambda x: x['user_id']).distinct()\
        .zipWithIndex().collectAsMap()
    
    business_users = in_file.map(lambda x: (x['business_id'], users[x['user_id']]))\
        .groupByKey().mapValues(list).persist()

    return business_users, len(users)

def min_hash(user_set, b_vals, a_vals, m, n_hash):
    sig = []
    for i in range(n_hash):
        find_min = list(map(lambda x: ((a_vals[i] * x + b_vals[i]) % m), user_set))
        sig.append(min(find_min))
    return sig

def lsh(signature, n_bands, n_rows):
    business_id, sig = signature
    buckets = []
    for i in range(n_bands):
        band = sig[i * n_rows: (i + 1) * n_rows]
        buckets.append(((i, tuple(band)), [business_id]))
    return buckets

def main(input_file, output_file, jac_thr, n_bands, n_rows, sc):
    n_hash = n_bands * n_rows    # number of hash functions we need 
    b_vals = random.sample(range(10000), n_hash)
    a_vals = random.sample(range(10000), n_hash)


    matrix, m = create_sig(sc, input_file)
    sig_matrix = matrix.mapValues(lambda x: min_hash(x, b_vals, a_vals, m, n_hash)).persist()
    lsh_bands = sig_matrix.flatMap(lambda x: lsh(x, n_bands, n_rows))\
        .reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)\
        .flatMap(lambda x: itertools.combinations(x[1], 2)).distinct()\
        .persist()
    
    s = matrix.collectAsMap()
    jac_sim = lsh_bands.map(lambda x: jaccard_sim(s, x))\
        .filter(lambda x: x[2] >= jac_thr).map(lambda x: {'b1': x[0], 'b2': x[1], 'sim': x[2]}).collect()
        
    with open(output_file, 'w') as outfile:
        for pair in jac_sim:
            outfile.write(json.dumps(pair) + '\n')
    


if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    input_file = 'data/train_review.json'
    output_file = 'rating.model'


    threshold = 0.4
    n_bands = 100
    n_rows = 2

    main(input_file, output_file, threshold, n_bands, n_rows, sc)

    sc.stop()