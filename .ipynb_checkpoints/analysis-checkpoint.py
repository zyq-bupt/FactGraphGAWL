
from data_loader_saver import load_data
import json
from tqdm import tqdm

    

def analysis_samples():

    hum_larger_cand_docid_list = []
    

    factgraph_output_file_path = "/root/autodl-fs/zyq/DeFacto/data/graph_similarity_result/merged_data1.json"
    all_data = load_data(factgraph_output_file_path)

    for graph_data in tqdm(all_data):
        intrinsic_error = graph_data['intrinsic_error']
        extrinsic_error = graph_data['extrinsic_error']

        if intrinsic_error == True and extrinsic_error == False:
            # print(graph_data['doc_id'])
            
            hum_score = graph_data['graph_sim']['article_humman_summary']
            cand_score = graph_data['graph_sim']['article_candidate']
            
            if hum_score > cand_score:
                hum_larger_cand_docid_list.append(graph_data['doc_id'])



    factgraph_output_file_path = '/root/autodl-fs/zyq/DeFacto/data/merged_data1_evaluate.json'

    # summary_type_list = ['abstract_score','candidate_score','humman_summary_score']
    fact_score_list=['cloze','dae_doc','factcc','summacconv','quals','feqa']
    with open(factgraph_output_file_path, 'r', encoding='utf-8') as f1:
        fs_data = json.load(f1)
        
    fs_docid_list = {'cloze':[],'dae_doc':[],'factcc':[],'summacconv':[],'quals':[],'feqa':[]}

    for item in fs_data:
        intrinsic_error = item['intrinsic_error']
        extrinsic_error = item['extrinsic_error']
        if intrinsic_error ==True and extrinsic_error == False:
            for fs in fact_score_list:
                
                hum_score = item['humman_summary_score'][fs]
                cand_score = item['candidate_score'][fs]
                
                if hum_score > cand_score:
                    fs_docid_list[fs].append(item['doc_id'])

    
    # for key in fs_docid_list.keys():
    #     print(key)
    #     docid_list = fs_docid_list[key]
        # difference_1 = list(set(hum_larger_cand_docid_list) - set(docid_list))
    common_elements = list(set(fs_docid_list['cloze']) & set(fs_docid_list['dae_doc'])& set(fs_docid_list['summacconv'])& set(fs_docid_list['quals'])& set(fs_docid_list['feqa']))
    difference_2 = list(set(common_elements) - set(hum_larger_cand_docid_list))
        # print(difference_1)
    print(difference_2)



# factgraph()
analysis_samples()