import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import wilcoxon
import scikit_posthocs as sp
from scipy.stats import shapiro
from common.data_loader_saver import load_data
import json
from tqdm import tqdm
import logging
import sys


# Step 2: 创建一个类来重定向 print() 的输出
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        # 只记录非空的消息
        if message.strip() != "":
            self.level(message)

    def flush(self):
        pass  # 此方法保持空即可


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
    if 'article_abstract' in data.keys():
        draw_df = {'reference':df['article_abstract'],
                'candidate':df['article_candidate'],
                    'human-generated':df['article_humman_summary']}
    else:
        draw_df = {'reference':df['abstract_score'],
                'candidate':df['candidate_score'],
                    'human-generated':df['humman_summary_score']}
    # 设置 Seaborn 样式
    sns.set(style="whitegrid")

    # 绘制箱形图
    plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df, palette='viridis')
    sns.boxplot(data=draw_df, palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # plt.xlabel('three types of summary')
    if fs == 'factgraph in intrinsic error' or fs == 'factgraph in extrinsic error':
        plt.title('WL-FSP Sample Distribution ')
    elif fs == 'summacconv':
        plt.title('SummaC Sample Distribution')
    
    elif fs == 'quals':
        plt.title('QUALs Sample Distribution')
    elif fs == 'feqa':
        plt.title('FEQA Sample Distribution')

    elif fs == 'factcc':
        plt.title('FactCC Sample Distribution')

    elif fs == 'dae_doc':
        plt.title('DAE Sample Distribution')

    elif fs == 'cloze':
        plt.title('ClozE Sample Distribution')
    

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
    #画箱线图
    draw_fig(fs, data, save_path)
    graphsim_list = ['article_abstract', 'article_candidate', 'article_humman_summary']
    #判断每组数据是否服从正态假设
    for gs in graphsim_list:
        d = data[gs]
        stat_shapiro, p_shapiro = shapiro(d)
        print(f"Statistic: {stat_shapiro}, P-value: {p_shapiro}")
        if p_shapiro > 0.05:
            print(gs+"数据服从正态分布 (p_shapiro > 0.05)")
        else:
            print(gs+"数据不服从正态分布 (p_shapiro <= 0.05)")
    print()
    #计算中位数和四分位数
    # 计算中位数
    median_list = []
    q1_list = []
    q3_list = []
    iqr_list = []
    for d in [data1,data2,data3]:
        median_list.append(np.median(d))
        # 计算四分位数 Q1 和 Q3
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        q1_list.append(q1)
        q3_list.append(q3)
        # 计算四分位距 IQR
        iqr_list.append(q3 - q1)

    print(f"中位数: {median_list}")
    print(f"下四分位数 (Q1): {q1_list}")
    print(f"上四分位数 (Q3): {q3_list}")
    print(f"四分位距 (IQR): {iqr_list}")
    print()

    # 判断非正态分布数据的显著性
    stat_friedm, p_friedm = friedmanchisquare(data1, data2, data3)
    print(f"Friedman Statistic: {stat_friedm}")
    print(f"P-value: {p_friedm}")

    if p_friedm > 0.05:
        print("三组之间没有显著差异 (p_friedm > 0.05)")
    else:
        print("三组之间存在显著差异 (p_friedm <= 0.05)")
    print()

    #判断两两之间的显著性
    stat12, p12 = wilcoxon(data1, data2)
    print(f"Wilcoxon Statistic: {stat12}")
    print(f"P-value: {p12}")
    if p12 > 0.05:
        print("1,2两组之间没有显著差异 (p > 0.05)")
    else:
        print("1,2两组之间存在显著差异 (p <= 0.05)")

    stat13, p13 = wilcoxon(data1, data3)
    print(f"Wilcoxon Statistic: {stat13}")
    print(f"P-value: {p13}")
    if p13 > 0.05:
        print("1,3两组之间没有显著差异 (p > 0.05)")
    else:
        print("1,3两组之间存在显著差异 (p <= 0.05)")

    stat23, p23 = wilcoxon(data2, data3)
    print(f"Wilcoxon Statistic: {stat23}")
    print(f"P-value: {p23}")
    if p23 > 0.05:
        print("2,3两组之间没有显著差异 (p > 0.05)")
    else:
        print("2,3两组之间存在显著差异 (p <= 0.05)")
    

def evaluate_factscore( data, save_path, fact_score_list):

    for fs in fact_score_list:
        all_type_summary_score = data[fs]
        print('----'*30)
        print(fs)

        for k, v in all_type_summary_score.items():
            all_type_summary_score[k] = np.array(v)
        data1 = all_type_summary_score['abstract_score']
        data2 = all_type_summary_score['candidate_score']
        data3 = all_type_summary_score['humman_summary_score']
        #画箱线图
        draw_fig(fs, all_type_summary_score, save_path)

        graphsim_list = ['abstract_score', 'candidate_score', 'humman_summary_score']
        #判断每组数据是否服从正态假设
        for gs in graphsim_list:
            d = all_type_summary_score[gs]
            stat_shapiro, p_shapiro = shapiro(d)
            print(f"Statistic: {stat_shapiro}, P-value: {p_shapiro}")
            if p_shapiro > 0.05:
                print(gs+"数据服从正态分布 (p_shapiro > 0.05)")
            else:
                print(gs+"数据不服从正态分布 (p_shapiro <= 0.05)")
        print('^^^^'*10)

        #计算中位数和四分位数
        # 计算中位数
        median_list = []
        q1_list = []
        q3_list = []
        iqr_list = []
        for d in [data1,data2,data3]:
            median_list.append(np.median(d))
            # 计算四分位数 Q1 和 Q3
            q1 = np.percentile(d, 25)
            q3 = np.percentile(d, 75)
            q1_list.append(q1)
            q3_list.append(q3)
            # 计算四分位距 IQR
            iqr_list.append(q3 - q1)

        print(f"中位数: {median_list}")
        print(f"下四分位数 (Q1): {q1_list}")
        print(f"上四分位数 (Q3): {q3_list}")
        print(f"四分位距 (IQR): {iqr_list}")
        print('^^^^'*10)

        # 判断非正态分布数据的显著性
        stat_friedm, p_friedm = friedmanchisquare(data1, data2, data3)
        print(f"Friedman Statistic: {stat_friedm}")
        print(f"P-value: {p_friedm}")

        if p_friedm > 0.05:
            print("三组之间没有显著差异 (p_friedm > 0.05)")
        else:
            print("三组之间存在显著差异 (p_friedm <= 0.05)")
        print('^^^^'*10)

        #判断两两之间的显著性
        stat12, p12 = wilcoxon(data1, data2)
        print(f"Wilcoxon Statistic: {stat12}")
        print(f"P-value: {p12}")
        if p12 > 0.05:
            print("1,2两组之间没有显著差异 (p > 0.05)")
        else:
            print("1,2两组之间存在显著差异 (p <= 0.05)")

        stat13, p13 = wilcoxon(data1, data3)
        print(f"Wilcoxon Statistic: {stat13}")
        print(f"P-value: {p13}")
        if p13 > 0.05:
            print("1,3两组之间没有显著差异 (p > 0.05)")
        else:
            print("1,3两组之间存在显著差异 (p <= 0.05)")

        stat23, p23 = wilcoxon(data2, data3)
        print(f"Wilcoxon Statistic: {stat23}")
        print(f"P-value: {p23}")
        if p23 > 0.05:
            print("2,3两组之间没有显著差异 (p > 0.05)")
        else:
            print("2,3两组之间存在显著差异 (p <= 0.05)")



def factgraph(save_path, factgraph_output_file_path):

    
    text_list=['abstract','candidate','humman_summary']
    graph_sims_ins = {}
    graph_sims_ext = {}
    hum_larger_than_cand_ins = 0
    count_ins = 0
    hum_larger_than_cand_ext = 0
    count_ext = 0
    
    
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
                    # print(graph_data['doc_id'])
                else:
                    graph_sims_ins[key].append(graph_data['graph_sim'][key])

            if graph_data['graph_sim']['article_humman_summary'] > graph_data['graph_sim']['article_candidate']:
                hum_larger_than_cand_ins += 1
            else:
                pass
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
    evaluate_factgraph('factgraph in intrinsic error', graph_sims_ins, save_path+'intrinsic/')
    
    print('********外部错误********')
    evaluate_factgraph('factgraph in extrinsic error', graph_sims_ext, save_path+'extrinsic/')

def otherfactscore(save_path):

    factgraph_output_file_path = '/root/autodl-fs/zyq/DeFacto/data/merged_data1_evaluate.json'
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

    print('********内部错误********')
    evaluate_factscore(fact_summary_score_ins, save_path+'intrinsic/', fact_score_list)
    print('----'*30)
    print('********外部错误********')
    evaluate_factscore(fact_summary_score_ext, save_path+'extrinsic/', fact_score_list)



if __name__ == '__main__':
    
    save_path = '/root/autodl-fs/zyq/DeFacto/data/fig_1_3/'
    # # Step 1: 配置日志记录器
    # logging.basicConfig(
    #     filename=save_path+'factgraph_evaluation_1_3.log',                # 日志文件名
    #     level=logging.INFO,                # 设置日志级别为 INFO
    #     format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    # )
    # # Step 3: 重定向标准输出到日志文件
    # sys.stdout = LoggerWriter(logging.info)
    
    # factgraph_output_file_path = "/root/autodl-fs/zyq/DeFacto/data/merged_data1.json"
    # factgraph(save_path, factgraph_output_file_path)#用于评估事实图结果

    # sys.stdout = sys.__stdout__
    # # 检查日志文件内容
    # print(f"日志已保存到 {save_path} 文件中，请查看以了解详细内容。")



    logging.basicConfig(
        filename=save_path+'factscores_evaluation.log',                # 日志文件名
        level=logging.INFO,                # 设置日志级别为 INFO
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )
    sys.stdout = LoggerWriter(logging.info)

    otherfactscore(save_path)#用于评估factscores

    sys.stdout = sys.__stdout__
    print(f"日志已保存到 {save_path} 文件中，请查看以了解详细内容。")