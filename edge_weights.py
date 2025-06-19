# 大类type，4种 token-token、token-mention、mention-entity、entity-entity 
edge_major_type_weights = {
  "token-token": 1,
  "token-mention":2,
  "mention-entity":4,
  "entity-entity ":1.5,
  # … 你还有哪些类型就在这里加上
}

# 小类type，
edge_minor_type_weights = {
  "nsubj":5,
  "obj":3,
  "amod":2,
  "RELATED_TO": 5,
  "BELONGS_TO": 5,
  "P17":6,
    "P27":6,
    "P54":6,
    "P118":6,
    "P131":6,
    "P102":6,
    "P937":6,
    "P150":6,
    "P102":6,
    "P488":6,
    # … 你还有哪些类型就在这里加上
}
#"nsubj":6,
 # "obj":5.5,
    # "amod":5,
    # "advmod":4.5,
    # "nmod":4,
    # "acl":3,
    # "xcomp":2,
    # "ccomp":1.5
 #    "P":1.5,
 #    "":1.5,
 #    "":1.5,
  # "RELATED_TO": 4,
    # "BELONGS_TO": 4,
    # "P17":6,
    # "P27":6,
    # "P54":6,
    # "P118":6,
    # "P131":6,
    # "P102":6,
    # "P937":6,
    # "P150":6,
    # "P102":6,
    # "P488":6,
