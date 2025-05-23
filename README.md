
# Automação de Conciliação Bancária com IA

Ferramenta para automatizar a conciliação bancária, classificando transações e sugerindo lançamentos contábeis utilizando inteligência artificial.

## Funcionalidades

- **Classificação automática de transações bancárias**
- **Sugestão de lançamentos contábeis**
- **Interface gráfica intuitiva**
- **Exportação de resultados em CSV/Excel**

## Como usar

1. **Clone o repositório**
   ```
   git clone https://github.com/seu-usuario/conciliacao-bancaria-ia.git
   ```
2. **Acesse o diretório do projeto**
   ```
   cd conciliacao-bancaria-ia
   ```
3. **Instale as dependências**
   ```
   pip install -r requirements.txt
   ```
   *(Se não tiver um arquivo `requirements.txt`, instale manualmente:)*
   ```
   pip install pandas scikit-learn
   ```
4. **Execute o arquivo principal**
   ```
   python conciliacao.py
   ```
5. **Na interface gráfica, selecione o extrato bancário** (ex: `extrato.csv`)
6. **Se desejar, selecione um arquivo de treinamento** (opcional)
7. **Verifique as sugestões de lançamentos contábeis**
8. **Exporte os resultados** para CSV ou Excel

## Requisitos

- **Python 3.8+**
- **pandas**
- **scikit-learn**
- *(adicione outros requisitos conforme necessário)*

## Exemplo de uso

Após instalar as dependências, execute:
```
python conciliacao.py
```
Na interface, selecione o arquivo do extrato bancário e, se desejar, o arquivo de treinamento. Ao final, um arquivo `resultados.csv` será gerado com as classificações e sugestões de lançamentos.

## Melhorias futuras

- **Integração com sistemas contábeis**
- **Suporte a mais formatos de extrato bancário**
- **Treinamento personalizado do modelo de IA**

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.
