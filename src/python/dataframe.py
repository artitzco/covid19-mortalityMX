import pandas as pd


class Catalogue:
    def __init__(self):
        self._catalogue = []

    def add(self, column, name=None, **kwargs):
        name = column if name is None else name
        self._catalogue.append(dict(column=column, name=name, **kwargs))

    def get(self):
        return self._catalogue


def filter(data, catalogue):
    subdata = pd.DataFrame(index=data.index)
    for dic in (catalogue.get() if isinstance(catalogue, Catalogue) else catalogue):
        column = dic['column']
        name = dic['name']
        if column in data:
            if 'value' in dic:
                subdata[name] = float('nan')
                for key, value in dic['value'].items():
                    subdata.loc[data[column] == key, name] = value
            elif 'category' in dic:
                subdata[name] = 'NE'
                for key, value in dic['category'].items():
                    subdata.loc[data[column] == key, name] = value
            elif 'function' in dic:
                if 'exception' in dic:
                    include = ~(data[column] == dic['exception'])
                    subdata[name] = pd.Series(
                        dic['default'], index=data.index)
                    subdata.loc[include, name] = dic['function'](
                        data[column][include])
                else:
                    subdata[name] = dic['function'](data[column])
            else:
                subdata[name] = data[column]
    return subdata