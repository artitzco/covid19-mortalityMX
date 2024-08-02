from scipy.stats import shapiro, levene, f_oneway, kruskal, ttest_ind
import pandas as pd
import pickle
import json
import os


def save_json(dict, file):
    with open(file, 'w', encoding='utf-8') as file:
        json.dump(dict, file, indent=4)


def load_json(file):
    with open(file, 'r') as file:
        return json.load(file)


def save_object(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object


def testing(name, sample, seed, function_name, category_name, category, function_do, category_do, session_do, path=''):
    summary_file = os.path.join(path, f'{name}_summary.json')
    save_json({fun: {cat: {'fmin': float('inf')}
                     for cat in ['best'] + category_name}
               for fun in function_name}, summary_file)
    for fun in function_name:
        function_do(fun)
        for cat in category_name:
            category_do(fun, cat)
            history_file = os.path.join(path, f'{name}_{cat}_{fun}.pkl')
            save_object([], history_file)
            for i in range(sample):
                session = session_do(seed+i, fun, cat)
                save_object(load_object(history_file) +
                            [session.history], history_file)
                summary = load_json(summary_file)
                if session.fmin < summary[fun][cat]['fmin']:
                    summary[fun][cat] = dict(seed=seed+i,
                                             time=session.time,
                                             niter=session.niter,
                                             fmin=session.fmin,
                                             xmin=session.xmin)
                    if session.fmin < summary[fun]['best']['fmin']:
                        summary[fun]['best'] = {category: cat}
                        summary[fun]['best'].update(summary[fun][cat])
                save_json(summary, summary_file)
                print(f'{fun} {cat} {round(100 * (i + 1) / sample)}%')
                del session
                del summary

    super_data = []
    for fun in function_name:
        for cat in category_name:
            history_file = os.path.join(path, f'{name}_{cat}_{fun}.pkl')
            data_list = load_object(history_file)
            os.remove(history_file)
            for i, data in enumerate(data_list):
                data['function'] = fun
                data[category] = cat
                data['seed'] = seed + i
            super_data += data_list
    save_object(
        pd.concat(super_data, axis=0), os.path.join(path, f'{name}.pkl'))


def stat_test(data, alpha=0.05):
    normal = True
    for col in data.columns:
        if shapiro(data[col])[1] < alpha:
            normal = False
            break
    cols = [data[col] for col in data.columns]
    if normal:
        if levene(*cols)[1] < alpha:
            pvalue = 1
            for i, di in enumerate(cols):
                for j, dj in enumerate(cols):
                    if i < j:
                        pvalue = min(pvalue,
                                     ttest_ind(di, dj, equal_var=False)[1])
            return 'Welch', pvalue
        return 'ANOVA', f_oneway(*cols)[1]
    return 'Kruskal-Wallis', kruskal(*cols)[1]


def compare_data(data, alpha=0.05):
    data = data[data.mean().sort_values().index]
    history = []
    while len(data.columns) > 1:
        test, pvalue = stat_test(data, alpha=alpha)
        history.append(('+'.join(data.columns), test, pvalue))
        if pvalue >= alpha:
            break
        data = data[data.columns[~(data.columns == data.mean().idxmax())]]
    return history, list(data.columns)
