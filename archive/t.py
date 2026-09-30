import requests
import urllib.parse
import urllib.request
import json

# DBpedia Spotlight 调用函数
def call_dbpedia_spotlight(text):
    dbpedia_entities = []
    response = requests.get(
        "https://api.dbpedia-spotlight.org/en/annotate", 
        params={"text": text},
        headers={"Accept": "application/json"}
    )
    if response.status_code == 200:
        data = response.json()
        for entity in data.get('Resources', []):
            entity_name = entity['@surfaceForm']
            dbpedia_entities.append(entity_name)
            dbpedia_url = entity['@URI']
            print(f"DBpedia Entity: {entity_name}, URI: {dbpedia_url}")
    else:
        print(f"DBpedia Spotlight API failed with status code: {response.status_code}")
    
    return dbpedia_entities

# Wikifier 调用函数
def call_wikifier(text, lang="en", threshold=1):
    wikifier_entities = []
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "tvuigkwpzwhskzmgibvgpudhdxjwhp"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
            for annotation in response.get("annotations", []):
                entity_name = annotation["title"]
                word_list=[]
                for s in annotation['support']:
                    start=s['chFrom']
                    end=s['chTo']
                    word_list.append(text[start:end+1])
                        
                print(word_list,annotation["title"], )
                wikifier_entities.append(word_list)
                # print(f"Wikifier Entity: {word}, URL: {annotation['url']}, support: {annotation['support']}")
                # print(f"Wikifier Entity: {word}, URL: {annotation['url']}, WikiData ID: {annotation['wikiDataItemId']}")
    except urllib.error.URLError as e:
        print(f"Request to Wikifier API failed: {e}")
    
    return wikifier_entities

# 筛选出在两个 API 中都存在的实体
def filter_common_entities(dbpedia_entities, wikifier_entities):

    # 使用集合的交集来获取两个实体列表中的公共实体
    common_entities = []
    
    # 遍历第一个列表的元素
    for element in dbpedia_entities:
        # 遍历第二个列表中的每一个子列表
        for sublist in wikifier_entities:
            # 如果当前元素在子列表中，说明找到了匹配项
            if element in sublist:
                common_entities.append(element)
                break  # 找到匹配后可以停止检查该元素
                
    print("\nCommon Entities Found in Both DBpedia Spotlight and Wikifier:")
    for entity in common_entities:
        print(f"Common Entity: {entity}")
    return common_entities

# 示例文本
text = """The charity said tests confirmed all of the cats near Victor Avenue, in Melton Mowbray, Leicestershire, had ingested the toxic substance. In the most recent case, seven-month-old Meereen died on Monday. An RSPCA spokesman said it was unclear whether the poisonings were accidental or deliberate. Updates on this story and more from LeicestershireThree other cats in the area have died in the last seven days, while another cat died two weeks ago. Meereen's "devastated" owner, Adria Pearce, said the cat came home on Friday evening and "seemed to be shivering a little". "I haven't been able to stop crying since she died," she said. "We found her behind the sofa, where she was foaming from the mouth and trying to be sick. "Meereen was taken to the vets - where it was confirmed she had consumed antifreeze - and died three days later. RSPCA inspector, Andy Bostock, is appealing for everyone in the area to ensure pesticides and chemicals were stored safely. "We are very concerned," he said. "It is the time of year where people use antifreeze in their cars, so if you do, please make sure there are no leaks and any spills are cleaned up properly."""
# text="Administrators of the ACT test took the decision just hours before some 5,500 students were due to sit it. The ACT is one of two entrance exams available to international and domestic students wanting to go to a US college. This is not the first cheating scandal to hit the tests in East Asia. The other entrance exam - the SAT - was cancelled in South Korea in 2013 because some of the questions were leaked. The ACT test was due to be held at 56 test centres in both South Korea and Hong Kong on Saturday morning. The Associated Press said teachers at some of Seoul's private \"cram schools\" said they were not notified until about an hour before the students were due to sit the test. ACT Inc, an Iowa-based non-profit organisation that was operating the test, said it took the decision after receiving \"credible evidence that test materials intended for administration in these regions have been compromised\". The organisation said in a statement that all students would get a refund but would only be able to resit when the tests are held again in September."
 
# 获取 DBpedia Spotlight 和 Wikifier 的实体
dbpedia_entities = call_dbpedia_spotlight(text)
wikifier_entities = call_wikifier(text)

# 获取两个 API 中相同的实体
common_entities = filter_common_entities(dbpedia_entities, wikifier_entities)
