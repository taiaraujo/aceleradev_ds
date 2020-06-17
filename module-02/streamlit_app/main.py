import streamlit as st
import pandas as pd
import base64


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


def get_imputation(df):
    imputed_exploitation = pd.DataFrame(
        {'nomes': df.columns, 'tipos': df.dtypes, 'NA #': df.isna().sum(),
         'NA %': (df.isna().sum() / df.shape[0]) * 100})
    st.table(imputed_exploitation[imputed_exploitation['tipos'] != 'object']['NA %'])
    st.subheader('Dados Inputados faça download abaixo : ')
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)


def main():
    st.title('AceleraDev Data Science')
    st.subheader('Semana 2 - Pré-processamento de Dados em Python')

    file = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type='csv')

    if file is not None:
        st.subheader('Analisando os dados')
        df = pd.read_csv(file)
        st.markdown('**Número de linhas:**')
        st.markdown(df.shape[0])
        st.markdown('**Número de colunas:**')
        st.markdown(df.shape[1])
        st.markdown('**Visualizando o dataframe**')
        number = st.slider('Escolha o numero de colunas que deseja ver', min_value=1, max_value=20)
        st.dataframe(df.head(number))
        st.markdown('**Nome das colunas:**')
        st.markdown(list(df.columns))

        exploitation = pd.DataFrame({'names': df.columns, 'types': df.dtypes, 'NA #': df.isna().sum(),
                                     'NA %': (df.isna().sum() / df.shape[0]) * 100})

        st.markdown('**Contagem dos tipos de dados:**')
        st.write(exploitation.types.value_counts())
        st.markdown('**Nomes das colunas do tipo int64:**')
        st.markdown(list(exploitation[exploitation['types'] == 'int64']['names']))
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(exploitation[exploitation['types'] == 'float64']['names']))
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(exploitation[exploitation['types'] == 'object']['names']))
        st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
        st.table(exploitation[exploitation['NA #'] != 0][['tipos', 'NA %']])
        st.subheader('Inputaçao de dados númericos :')
        percentual = st.slider('Escolha o limite de percentual faltante limite para as colunas vocë '
                               'deseja inputar os dados', min_value=0, max_value=100)
        lista_colunas = list(exploitation[exploitation['NA %'] < percentual]['names'])
        select_method = st.radio('Escolha um metodo abaixo :', ('Mean', 'Median'))
        st.markdown('Você selecionou : ' + str(select_method))

        if select_method == 'Mean':
            df_imputed = df[lista_colunas].fillna(df[lista_colunas].mean())
            get_imputation(df_imputed)

        if select_method == 'Median':
            df_imputed = df[lista_colunas].fillna(df[lista_colunas].median())
            get_imputation(df_imputed)


if __name__ == '__main__':
    main()
