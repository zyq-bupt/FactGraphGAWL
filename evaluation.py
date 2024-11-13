import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from data_loader_saver import load_data
import json
from tqdm import tqdm

#方差分析检验三组数据的显著性，ANOVA
def anova(data1,data2,data3):
    # 进行单因素方差分析
    f_statistic, p_value = stats.f_oneway(data1,data2,data3)

    print("F统计量:", f_statistic)
    print("p值:", p_value)

    # 检验结果
    alpha = 0.05
    if p_value < alpha:
        print("拒绝零假设，至少有一组的均值与其他组显著不同。")
    else:
        print("无法拒绝零假设，各组的均值没有显著差异。")
    
    return f_statistic, p_value


def tukey_HSD(data1, data2, data3):
    # 将数据整理为DataFrame
    data = {
        'score': np.concatenate([data1, data2, data3]),
        'group': ['group1'] * len(data1) + ['group2'] * len(data2) + ['group3'] * len(data3)
    }
    df = pd.DataFrame(data)

    # 进行方差分析
    model = ols('score ~ group', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 进行Tukey HSD事后比较
    tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['group'], alpha=0.05)
    
    return anova_table, tukey

def draw_fig(fs, data, save_path):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 设置 Seaborn 样式
    sns.set(style="whitegrid")

    # 绘制箱形图
    plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df, palette='viridis')
    sns.boxplot(data=df, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
    # 添加标题和标签
    plt.title('Box Plot of Sample Distribution')
    plt.xlabel('three types of summary score')
    plt.ylabel('%s score'%(fs))
    plt.savefig(save_path+fs+'.jpg', dpi=None, bbox_inches='tight', pad_inches=0.1)
    # 显示图形
    plt.tight_layout()
    plt.show()

def evaluate_factgraph(fs, data, save_path):
    for k, v in data.items():
        data[k] = np.array(v)
        
    #     检查是否有NaN和None值  
    #     has_nan = np.isnan(v).any()
    #     print("数组中是否有NaN值:", has_nan,k)
    #     if has_nan:
    #         nan_indices = np.where(np.isnan(v))
    #         print("NaN 值的位置:", nan_indices,v[nan_indices])
    #     has_none = None in graph_sims[k]
    #     print("数组中是否有None值:", has_none,k)
    data1 = data['article_abstract']
    data2 = data['article_candidate']
    data3 = data['article_humman_summary']   
    draw_fig(fs, data, save_path)
    f, p = anova(data1, data2, data3)
    anova_table, tukey = tukey_HSD(data1, data2, data3)

    return f, p, anova_table, tukey 

def evaluate_factscore( data, save_path, fact_score_list):
    f_list = []
    p_list = []
    anova_table_list = []
    tukey_list = []
    hum_larger_than_cand_list = []

    for fs in fact_score_list:
        all_type_summary_score = data[fs]
        print(fs)
        
        hum_larger_than_cand = 0

        for k, v in all_type_summary_score.items():
            all_type_summary_score[k] = np.array(v)
        data1 = all_type_summary_score['abstract_score']
        data2 = all_type_summary_score['candidate_score']
        data3 = all_type_summary_score['humman_summary_score']

        draw_fig(fs, all_type_summary_score, save_path)

        f, p = anova(data1, data2, data3)
        anova_table, tukey = tukey_HSD(data1, data2, data3)
        hum_larger_than_cand = np.sum(data3 > data2)

        f_list.append(f)
        p_list.append(p)
        anova_table_list.append(anova_table)
        tukey_list.append(tukey)
        hum_larger_than_cand_list.append(hum_larger_than_cand)


    return f_list, p_list, anova_table_list, tukey_list, hum_larger_than_cand_list, len(data2)

def factgraph():

    save_path = '/root/autodl-fs/zyq/DeFacto/data/fig/'
    text_list=['abstract','candidate','humman_summary']
    graph_sims_ins = {}
    graph_sims_ext = {}
    hum_larger_than_cand_ins = 0
    count_ins = 0
    hum_larger_than_cand_ext = 0
    count_ext = 0

    factgraph_output_file_path = "/root/autodl-fs/zyq/DeFacto/data/graph_similarity_result/merged_data1.json"
    all_data = load_data(factgraph_output_file_path)

    for graph_data in tqdm(all_data):
        intrinsic_error = graph_data['intrinsic_error']
        extrinsic_error = graph_data['extrinsic_error']

        if intrinsic_error == True and extrinsic_error == False:
            key1 = 'article'
            # print(graph_data['doc_id'])
            for key2 in text_list:
                key = '%s_%s'%(key1,key2)
                if '%s_%s'%(key1,key2) not in graph_sims_ins.keys():
                    graph_sims_ins[key] = []

                has_nan = np.isnan(graph_data['graph_sim'][key])
                if has_nan:
                    # print("数组中有NaN值:", graph_data['doc_id'],len(graph_sims_ins[key]))
                    graph_sims_ins[key].append(0)
                else:
                    graph_sims_ins[key].append(graph_data['graph_sim'][key])

            if graph_data['graph_sim']['article_humman_summary'] > graph_data['graph_sim']['article_candidate']:
                hum_larger_than_cand_ins += 1
            count_ins += 1


        elif intrinsic_error == False and extrinsic_error == True:
            key1 = 'article'
            # print(graph_data['doc_id'])
            for key2 in text_list:
                key = '%s_%s'%(key1,key2)
                if '%s_%s'%(key1,key2) not in graph_sims_ext.keys():
                    graph_sims_ext[key] = []

                has_nan = np.isnan(graph_data['graph_sim'][key])
                if has_nan:
                    # print("数组中有NaN值:", graph_data['doc_id'],len(graph_sims_ins[key]))
                    graph_sims_ext[key].append(0)
                else:
                    graph_sims_ext[key].append(graph_data['graph_sim'][key])
            
            if graph_data['graph_sim']['article_humman_summary'] > graph_data['graph_sim']['article_candidate']:
                hum_larger_than_cand_ext += 1
            count_ext += 1


    print('********内部错误********')
    f, p, anova_table, tukey  = evaluate_factgraph('factgraph in intrinsic error', graph_sims_ins, save_path)
    ins_eva_list = [f, p, anova_table, tukey, hum_larger_than_cand_ins, count_ins]

    print('********外部错误********')
    fe, pe, anova_tablee, tukeye  = evaluate_factgraph('factgraph in extrinsic error', graph_sims_ext, save_path)
    ext_eva_list = [fe, pe, anova_tablee, tukeye, hum_larger_than_cand_ext, count_ext]
   
    # 写入文件
    with open('/root/autodl-fs/zyq/DeFacto/data/fig/factgraph_eval.txt', "w", encoding="utf-8") as file:
        # 写入 f_list
        file.write("ins:\n")
        for item in ins_eva_list:
            file.write(f"{item}\n")

        file.write("ext:\n")
        for item in ext_eva_list:
            file.write(f"{item}\n")

def otherfactscore():

    factgraph_output_file_path = '/root/autodl-fs/zyq/DeFacto/data/merged_data1_evaluate.json'
    save_path = '/root/autodl-fs/zyq/DeFacto/data/fig/'
    text_list=['abstract','candidate','humman_summary']
    summary_type_list = ['abstract_score','candidate_score','humman_summary_score']
    fact_score_list=['cloze','dae_doc','factcc','summacconv','quals','feqa']
    with open(factgraph_output_file_path, 'r', encoding='utf-8') as f1:
        graph_data = json.load(f1)
        
    fact_summary_score_ins = {}
    fact_summary_score_ext = {}
    for item in graph_data:
        intrinsic_error = item['intrinsic_error']
        extrinsic_error = item['extrinsic_error']
        if intrinsic_error ==True and extrinsic_error == False:
            for fs in fact_score_list:
                if fs not in fact_summary_score_ins.keys():
                    fact_summary_score_ins[fs]={}
                for st in summary_type_list:
                    factscore = item[st][fs]
                    if st not in fact_summary_score_ins[fs].keys():
                        fact_summary_score_ins[fs][st]=[]
                    fact_summary_score_ins[fs][st].append(factscore)
       
        
        elif intrinsic_error ==False and extrinsic_error == True:
            for fs in fact_score_list:
                if fs not in fact_summary_score_ext.keys():
                    fact_summary_score_ext[fs]={}
                for st in summary_type_list:
                    factscore = item[st][fs]
                    if st not in fact_summary_score_ext[fs].keys():
                        fact_summary_score_ext[fs][st]=[]
                    fact_summary_score_ext[fs][st].append(factscore)

    f_list, p_list, anova_table_list, tukey_list, hum_larger_than_cand_list, count = evaluate_factscore(fact_summary_score_ins, save_path, fact_score_list)
    # f_list, p_list, anova_table_list, tukey_list, hum_larger_than_cand_list, count = evaluate_factscore(fact_summary_score_ext, save_path, fact_score_list)

    # 写入文件
    with open('/root/autodl-fs/zyq/DeFacto/data/fig/instract/factscore_ext_ins.txt', "w", encoding="utf-8") as file:
        # 写入 f_list
        file.write("f_list:\n")
        for item, fs in zip(f_list, fact_score_list):
            file.write(fs+'\n')
            file.write(f"{item}\n")
        
        # 写入 p_list
        file.write("\np_list:\n")
        for item, fs in zip(p_list, fact_score_list):
            file.write(fs+'\n')
            file.write(f"{item}\n")
        
        # 写入 anova_table_list
        file.write("\nanova_table_list:\n")
        for item, fs in zip(anova_table_list, fact_score_list):
            file.write(fs)
            file.write(f"{item}\n")
        
        # 写入 tukey_list
        file.write("\ntukey_list:\n")
        for item, fs in zip(tukey_list, fact_score_list):
            file.write(fs+'\n')
            file.write(f"{item}\n")
        
        # 写入 hum_larger_than_cand_list
        file.write("\nhum_larger_than_cand_list:\n")
        for item, fs in zip(hum_larger_than_cand_list, fact_score_list):
            file.write(fs+'\n')
            file.write(f"{item}\n")



# factgraph()
otherfactscore()