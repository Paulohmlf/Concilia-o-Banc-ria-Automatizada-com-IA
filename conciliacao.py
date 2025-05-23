# =============================================================================
# INTERFACE GRÁFICA PARA SISTEMA DE CONCILIAÇÃO BANCÁRIA
# =============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
from datetime import datetime
import threading

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================
def configurar_logging():
    """Configura logging com suporte a UTF-8."""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(
        'conciliacao.log', 
        mode='a', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    
    return logger

logger = configurar_logging()

# =============================================================================
# CLASSE DO SISTEMA DE CONCILIAÇÃO (BACKEND)
# =============================================================================
class ConciliadorBancario:
    """Sistema automatizado de conciliação bancária usando Machine Learning."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._carregar_configuracao(config)
        self.vectorizer = None
        self.model = None
        self.mapa_contabil = self._definir_mapa_contabil()
        
        # Tentar carregar modelo automaticamente na inicialização
        self.carregar_modelo_salvo()
        
    def _carregar_configuracao(self, config: Optional[Dict] = None) -> Dict:
        """Carrega configurações do sistema."""
        configuracao_padrao = {
            'caminho_treino': None,
            'caminho_extrato': None,
            'caminho_modelo': Path('modelos/modelo_conciliacao.pkl'),
            'max_features': 1000,
            'n_estimators': 100,
            'random_state': 42,
            'test_size': 0.2
        }
        
        if config:
            configuracao_padrao.update(config)
            
        return configuracao_padrao
    
    def _definir_mapa_contabil(self) -> Dict:
        """Define o mapeamento de categorias para contas contábeis."""
        return {
            'Recebimento de Cliente': {
                'debito': '1.1.1.01 - Banco', 
                'credito': '3.1.1.01 - Receita de Vendas'
            },
            'Fornecedores': {
                'debito': '4.1.2.01 - Custo de Mercadorias/Serviços', 
                'credito': '1.1.1.01 - Banco'
            },
            'Taxas Bancárias': {
                'debito': '4.4.1.05 - Despesas Bancárias', 
                'credito': '1.1.1.01 - Banco'
            },
            'Folha de Pagamento': {
                'debito': '4.4.1.01 - Desp. com Salários', 
                'credito': '1.1.1.01 - Banco'
            },
            'Impostos': {
                'debito': '4.4.3.01 - Impostos e Taxas', 
                'credito': '1.1.1.01 - Banco'
            },
            'Aluguel': {
                'debito': '4.4.1.02 - Desp. com Aluguéis', 
                'credito': '1.1.1.01 - Banco'
            },
            'Despesas Gerais': {
                'debito': '4.4.1.99 - Outras Despesas Adm.', 
                'credito': '1.1.1.01 - Banco'
            }
        }
    
    def modelo_esta_carregado(self) -> bool:
        """Verifica se o modelo está carregado e pronto para uso."""
        return self.model is not None and self.vectorizer is not None
    
    def validar_arquivos(self) -> bool:
        """Valida se os arquivos necessários existem."""
        if not self.config['caminho_treino'] or not self.config['caminho_extrato']:
            return False
            
        arquivos_necessarios = [
            Path(self.config['caminho_treino']),
            Path(self.config['caminho_extrato'])
        ]
        
        for arquivo in arquivos_necessarios:
            if not arquivo.exists():
                logger.error(f"Arquivo não encontrado: {arquivo}")
                return False
                
        logger.info("Todos os arquivos necessários foram encontrados")
        return True
    
    def carregar_dados_treino(self) -> Tuple[pd.DataFrame, bool]:
        """Carrega e valida os dados de treinamento."""
        try:
            df_treino = pd.read_csv(self.config['caminho_treino'])
            
            colunas_necessarias = ['descricao', 'categoria']
            if not all(col in df_treino.columns for col in colunas_necessarias):
                logger.error(f"Colunas necessárias não encontradas: {colunas_necessarias}")
                return pd.DataFrame(), False
            
            df_treino = df_treino.dropna(subset=colunas_necessarias)
            df_treino['descricao'] = df_treino['descricao'].astype(str).str.strip()
            
            logger.info(f"Dados de treinamento carregados: {len(df_treino)} registros")
            return df_treino, True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de treinamento: {e}")
            return pd.DataFrame(), False
    
    def treinar_modelo(self, df_treino: pd.DataFrame, callback=None) -> bool:
        """Treina o modelo de classificação."""
        try:
            if callback:
                callback("Iniciando treinamento do modelo...")
            
            X = df_treino['descricao']
            y = df_treino['categoria']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state'],
                stratify=y
            )
            
            if callback:
                callback("Vetorizando textos...")
            
            self.vectorizer = TfidfVectorizer(
                max_features=self.config['max_features'],
                stop_words=None,
                ngram_range=(1, 2)
            )
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            if callback:
                callback("Treinando modelo Random Forest...")
            
            self.model = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                random_state=self.config['random_state'],
                class_weight='balanced'
            )
            
            self.model.fit(X_train_vec, y_train)
            
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            if callback:
                callback(f"Modelo treinado com sucesso! Acurácia: {accuracy:.2%}")
            
            self._salvar_modelo()
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")
            if callback:
                callback(f"Erro no treinamento: {e}")
            return False
    
    def _salvar_modelo(self):
        """Salva o modelo treinado."""
        try:
            self.config['caminho_modelo'].parent.mkdir(parents=True, exist_ok=True)
            
            modelo_dados = {
                'vectorizer': self.vectorizer,
                'model': self.model,
                'data_treino': datetime.now(),
                'config': self.config
            }
            
            joblib.dump(modelo_dados, self.config['caminho_modelo'])
            logger.info(f"Modelo salvo em: {self.config['caminho_modelo']}")
            
        except Exception as e:
            logger.warning(f"Não foi possível salvar o modelo: {e}")
    
    def carregar_modelo_salvo(self) -> bool:
        """Carrega um modelo previamente salvo."""
        try:
            if not self.config['caminho_modelo'].exists():
                logger.info("Arquivo de modelo não encontrado")
                return False
                
            modelo_dados = joblib.load(self.config['caminho_modelo'])
            self.vectorizer = modelo_dados['vectorizer']
            self.model = modelo_dados['model']
            
            logger.info("Modelo carregado do arquivo salvo com sucesso")
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo salvo: {e}")
            return False
    
    def processar_extrato(self, callback=None) -> Optional[pd.DataFrame]:
        """Processa o extrato bancário e gera sugestões de conciliação."""
        try:
            if callback:
                callback("Carregando extrato bancário...")
            
            df_extrato = pd.read_csv(self.config['caminho_extrato'])
            
            if 'descricao' not in df_extrato.columns:
                logger.error("Coluna 'descricao' não encontrada no extrato")
                return None
            
            df_extrato['descricao'] = df_extrato['descricao'].astype(str).str.strip()
            df_extrato = df_extrato.dropna(subset=['descricao'])
            
            if callback:
                callback("Classificando transações...")
            
            X_extrato_vec = self.vectorizer.transform(df_extrato['descricao'])
            previsoes = self.model.predict(X_extrato_vec)
            probabilidades = self.model.predict_proba(X_extrato_vec)
            
            df_extrato['categoria_sugerida'] = previsoes
            df_extrato['confianca'] = probabilidades.max(axis=1)
            df_extrato['sugestao_lancamento'] = df_extrato.apply(
                self._gerar_sugestao_lancamento, axis=1
            )
            
            if callback:
                callback(f"Extrato processado: {len(df_extrato)} transações")
            
            return df_extrato
            
        except Exception as e:
            logger.error(f"Erro ao processar extrato: {e}")
            if callback:
                callback(f"Erro ao processar extrato: {e}")
            return None
    
    def _gerar_sugestao_lancamento(self, row: pd.Series) -> str:
        """Gera sugestão de lançamento contábil para uma transação."""
        categoria = row['categoria_sugerida']
        confianca = row['confianca']
        
        if categoria in self.mapa_contabil:
            debito = self.mapa_contabil[categoria]['debito']
            credito = self.mapa_contabil[categoria]['credito']
            return f"D: {debito} | C: {credito} (Confiança: {confianca:.1%})"
        else:
            return f"Categoria não mapeada: {categoria} (Confiança: {confianca:.1%})"

# =============================================================================
# INTERFACE GRÁFICA PRINCIPAL
# =============================================================================
class ConciliadorGUI:
    """Interface gráfica para o sistema de conciliação bancária."""
    
    def __init__(self):
        # Criar janela principal
        self.root = tk.Tk()
        self.root.title("Sistema de Conciliação Bancária Automatizada")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configurar estilo
        self.configurar_estilo()
        
        # Variáveis
        self.conciliador = ConciliadorBancario()
        self.arquivo_treino = tk.StringVar()
        self.arquivo_extrato = tk.StringVar()
        self.df_resultado = None
        
        # Configurar interface
        self.criar_interface()
        self.centralizar_janela()
        
        # Verificar status do modelo na inicialização
        self.verificar_status_modelo()
    
    def configurar_estilo(self):
        """Configura o estilo da interface."""
        # Configurar cores
        self.cores = {
            'primary': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Configurar fonte padrão
        self.root.option_add('*Font', 'Arial 9')
        
        # Configurar estilo ttk
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar cores dos botões
        style.configure('Primary.TButton', background=self.cores['primary'])
        style.configure('Success.TButton', background=self.cores['success'])
        style.configure('Warning.TButton', background=self.cores['warning'])
        style.configure('Info.TButton', background=self.cores['info'])
    
    def verificar_status_modelo(self):
        """Verifica o status do modelo na inicialização."""
        if self.conciliador.modelo_esta_carregado():
            self.adicionar_log("🎉 Modelo carregado automaticamente! Sistema pronto para uso.")
            self.atualizar_status("✅ Modelo carregado - Pronto para processar")
        else:
            self.adicionar_log("⚠️ Nenhum modelo encontrado. Selecione dados de treinamento para treinar.")
            self.atualizar_status("⚠️ Modelo não encontrado - Treine primeiro")
    
    def verificar_modelo_disponivel(self):
        """Verifica se há modelo disponível (carregado ou para treinar)."""
        # Se já está carregado, ok
        if self.conciliador.modelo_esta_carregado():
            return True
        
        # Tentar carregar modelo salvo
        if self.conciliador.carregar_modelo_salvo():
            self.adicionar_log("✅ Modelo carregado automaticamente")
            return True
        
        # Se não tem modelo salvo e nem arquivo de treino, erro
        if not self.arquivo_treino.get():
            return False
        
        return True
        
    def centralizar_janela(self):
        """Centraliza a janela na tela."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def criar_interface(self):
        """Cria a interface gráfica principal."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        titulo = tk.Label(
            main_frame, 
            text="Sistema de Conciliação Bancária Automatizada",
            font=("Arial", 16, "bold"),
            fg=self.cores['primary']
        )
        titulo.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Seção de arquivos
        self.criar_secao_arquivos(main_frame)
        
        # Seção de controles
        self.criar_secao_controles(main_frame)
        
        # Seção de log
        self.criar_secao_log(main_frame)
        
        # Seção de resultados
        self.criar_secao_resultados(main_frame)
        
        # Barra de status
        self.criar_barra_status()
    
    def criar_secao_arquivos(self, parent):
        """Cria a seção de seleção de arquivos."""
        # Frame para arquivos
        arquivo_frame = ttk.LabelFrame(parent, text="Seleção de Arquivos", padding="10")
        arquivo_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        arquivo_frame.columnconfigure(1, weight=1)
        
        # Arquivo de treinamento
        ttk.Label(arquivo_frame, text="Dados de Treinamento (opcional):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(arquivo_frame, textvariable=self.arquivo_treino, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2
        )
        ttk.Button(
            arquivo_frame, 
            text="Procurar", 
            command=self.selecionar_arquivo_treino
        ).grid(row=0, column=2, pady=2)
        
        # Arquivo de extrato
        ttk.Label(arquivo_frame, text="Extrato Bancário:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(arquivo_frame, textvariable=self.arquivo_extrato, state="readonly").grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=2
        )
        ttk.Button(
            arquivo_frame, 
            text="Procurar", 
            command=self.selecionar_arquivo_extrato
        ).grid(row=1, column=2, pady=2)
    
    def criar_secao_controles(self, parent):
        """Cria a seção de controles."""
        # Frame para controles
        controle_frame = ttk.LabelFrame(parent, text="Controles", padding="10")
        controle_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Botões principais
        botoes_frame = ttk.Frame(controle_frame)
        botoes_frame.pack(fill=tk.X)
        
        self.btn_treinar = ttk.Button(
            botoes_frame, 
            text="🎯 Treinar Modelo", 
            command=self.treinar_modelo,
            style="Success.TButton"
        )
        self.btn_treinar.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_processar = ttk.Button(
            botoes_frame, 
            text="⚡ Processar Extrato", 
            command=self.processar_extrato,
            style="Primary.TButton"
        )
        self.btn_processar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botões de exportação
        self.btn_exportar_csv = ttk.Button(
            botoes_frame, 
            text="📄 Exportar CSV", 
            command=self.exportar_csv,
            style="Info.TButton",
            state="disabled"
        )
        self.btn_exportar_csv.pack(side=tk.LEFT, padx=(0, 5))
        
        self.btn_exportar_excel = ttk.Button(
            botoes_frame, 
            text="📊 Exportar Excel", 
            command=self.exportar_excel,
            style="Info.TButton",
            state="disabled"
        )
        self.btn_exportar_excel.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            botoes_frame, 
            text="🗑️ Limpar Log", 
            command=self.limpar_log,
            style="Warning.TButton"
        ).pack(side=tk.RIGHT)
        
        # Barra de progresso
        self.progress = ttk.Progressbar(controle_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def criar_secao_log(self, parent):
        """Cria a seção de log."""
        # Frame para log
        log_frame = ttk.LabelFrame(parent, text="Log do Sistema", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        
        # Área de texto para log
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            height=8, 
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg='#f8f9fa'
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Adicionar mensagem inicial
        self.adicionar_log("🚀 Sistema iniciado. Verificando modelo salvo...")
    
    def criar_secao_resultados(self, parent):
        """Cria a seção de resultados."""
        # Frame para resultados
        resultado_frame = ttk.LabelFrame(parent, text="Resultados da Conciliação", padding="10")
        resultado_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        resultado_frame.columnconfigure(0, weight=1)
        resultado_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(4, weight=2)
        
        # Treeview para resultados
        colunas = ("descricao", "categoria_sugerida", "confianca", "sugestao_lancamento")
        self.tree_resultados = ttk.Treeview(resultado_frame, columns=colunas, show="headings", height=10)
        
        # Configurar cabeçalhos
        self.tree_resultados.heading("descricao", text="Descrição")
        self.tree_resultados.heading("categoria_sugerida", text="Categoria")
        self.tree_resultados.heading("confianca", text="Confiança")
        self.tree_resultados.heading("sugestao_lancamento", text="Sugestão de Lançamento")
        
        # Configurar larguras
        self.tree_resultados.column("descricao", width=300)
        self.tree_resultados.column("categoria_sugerida", width=150)
        self.tree_resultados.column("confianca", width=100)
        self.tree_resultados.column("sugestao_lancamento", width=400)
        
        # Scrollbars
        scrollbar_v = ttk.Scrollbar(resultado_frame, orient=tk.VERTICAL, command=self.tree_resultados.yview)
        scrollbar_h = ttk.Scrollbar(resultado_frame, orient=tk.HORIZONTAL, command=self.tree_resultados.xview)
        self.tree_resultados.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Grid
        self.tree_resultados.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def criar_barra_status(self):
        """Cria a barra de status."""
        self.status_var = tk.StringVar()
        self.status_var.set("🔄 Inicializando...")
        
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg=self.cores['light'],
            padx=5,
            pady=2
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def selecionar_arquivo_treino(self):
        """Seleciona o arquivo de dados de treinamento."""
        arquivo = filedialog.askopenfilename(
            title="Selecionar arquivo de treinamento",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if arquivo:
            self.arquivo_treino.set(arquivo)
            self.conciliador.config['caminho_treino'] = arquivo
            self.adicionar_log(f"📁 Arquivo de treinamento selecionado: {arquivo}")
    
    def selecionar_arquivo_extrato(self):
        """Seleciona o arquivo de extrato bancário."""
        arquivo = filedialog.askopenfilename(
            title="Selecionar extrato bancário",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if arquivo:
            self.arquivo_extrato.set(arquivo)
            self.conciliador.config['caminho_extrato'] = arquivo
            self.adicionar_log(f"📄 Extrato bancário selecionado: {arquivo}")
    
    def adicionar_log(self, mensagem):
        """Adiciona uma mensagem ao log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {mensagem}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def limpar_log(self):
        """Limpa o log."""
        self.log_text.delete(1.0, tk.END)
        self.adicionar_log("🧹 Log limpo")
    
    def atualizar_status(self, mensagem):
        """Atualiza a barra de status."""
        self.status_var.set(mensagem)
        self.root.update_idletasks()
    
    def treinar_modelo(self):
        """Treina o modelo em uma thread separada."""
        if not self.arquivo_treino.get():
            messagebox.showerror("Erro", "Selecione o arquivo de dados de treinamento!")
            return
        
        def callback_treino(mensagem):
            self.root.after(0, lambda: self.adicionar_log(mensagem))
        
        def executar_treino():
            try:
                self.root.after(0, lambda: self.progress.start())
                self.root.after(0, lambda: self.atualizar_status("🔄 Treinando modelo..."))
                
                # Sempre treinar novo modelo quando solicitado
                callback_treino("🔄 Iniciando novo treinamento...")
                
                df_treino, sucesso = self.conciliador.carregar_dados_treino()
                if not sucesso:
                    callback_treino("❌ Erro ao carregar dados de treinamento")
                    return
                
                if not self.conciliador.treinar_modelo(df_treino, callback_treino):
                    callback_treino("❌ Erro no treinamento do modelo")
                    return
                
                callback_treino("🎉 Modelo treinado e salvo com sucesso!")
                self.root.after(0, lambda: self.atualizar_status("✅ Modelo treinado - Pronto para processar"))
                
            except Exception as e:
                callback_treino(f"❌ Erro inesperado: {e}")
            finally:
                self.root.after(0, lambda: self.progress.stop())
                if not self.conciliador.modelo_esta_carregado():
                    self.root.after(0, lambda: self.atualizar_status("❌ Erro no treinamento"))
        
        # Executar em thread separada
        thread = threading.Thread(target=executar_treino)
        thread.daemon = True
        thread.start()
    
    def processar_extrato(self):
        """Processa o extrato em uma thread separada."""
        if not self.arquivo_extrato.get():
            messagebox.showerror("Erro", "Selecione o arquivo de extrato bancário!")
            return
        
        # Verificar se modelo está disponível
        if not self.verificar_modelo_disponivel():
            messagebox.showerror(
                "Erro", 
                "Nenhum modelo encontrado!\n\n" +
                "Selecione os dados de treinamento e clique em 'Treinar Modelo' primeiro."
            )
            return
        
        def callback_processamento(mensagem):
            self.root.after(0, lambda: self.adicionar_log(mensagem))
        
        def executar_processamento():
            try:
                self.root.after(0, lambda: self.progress.start())
                self.root.after(0, lambda: self.atualizar_status("⚡ Processando extrato..."))
                
                self.df_resultado = self.conciliador.processar_extrato(callback_processamento)
                
                if self.df_resultado is not None:
                    self.root.after(0, self.atualizar_resultados)
                    callback_processamento("🎉 Processamento concluído com sucesso!")
                else:
                    callback_processamento("❌ Erro no processamento do extrato")
                
            except Exception as e:
                callback_processamento(f"❌ Erro inesperado: {e}")
            finally:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.atualizar_status("✅ Pronto"))
        
        # Executar em thread separada
        thread = threading.Thread(target=executar_processamento)
        thread.daemon = True
        thread.start()
    
    def atualizar_resultados(self):
        """Atualiza a tabela de resultados."""
        # Limpar resultados anteriores
        for item in self.tree_resultados.get_children():
            self.tree_resultados.delete(item)
        
        if self.df_resultado is not None:
            # Adicionar novos resultados
            for index, row in self.df_resultado.iterrows():
                valores = (
                    row['descricao'][:50] + "..." if len(str(row['descricao'])) > 50 else row['descricao'],
                    row['categoria_sugerida'],
                    f"{row['confianca']:.1%}",
                    row['sugestao_lancamento']
                )
                
                # Colorir linhas com baixa confiança
                item = self.tree_resultados.insert("", "end", values=valores)
                if row['confianca'] < 0.7:
                    self.tree_resultados.set(item, "confianca", f"{row['confianca']:.1%} ⚠️")
            
            # Habilitar botões de exportar
            self.btn_exportar_csv.config(state="normal")
            self.btn_exportar_excel.config(state="normal")
            
            # Mostrar estatísticas
            total = len(self.df_resultado)
            baixa_confianca = len(self.df_resultado[self.df_resultado['confianca'] < 0.7])
            confianca_media = self.df_resultado['confianca'].mean()
            
            self.adicionar_log(f"📊 Resultados: {total} transações processadas")
            self.adicionar_log(f"🎯 Confiança média: {confianca_media:.1%}")
            if baixa_confianca > 0:
                self.adicionar_log(f"⚠️ {baixa_confianca} transações com baixa confiança")
    
    def exportar_csv(self):
        """Exporta os resultados para CSV."""
        if self.df_resultado is None:
            messagebox.showerror("Erro", "Não há resultados para exportar!")
            return
        
        arquivo = filedialog.asksaveasfilename(
            title="Salvar resultados em CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if arquivo:
            try:
                # Exportar para CSV com encoding UTF-8
                self.df_resultado.to_csv(arquivo, index=False, encoding='utf-8-sig')
                self.adicionar_log(f"📄 Resultados exportados para CSV: {arquivo}")
                messagebox.showinfo("Sucesso", "Resultados exportados em CSV com sucesso!")
            except Exception as e:
                self.adicionar_log(f"❌ Erro ao exportar CSV: {e}")
                messagebox.showerror("Erro", f"Erro ao exportar para CSV: {e}")

    def exportar_excel(self):
        """Exporta os resultados para Excel (.xlsx)."""
        if self.df_resultado is None:
            messagebox.showerror("Erro", "Não há resultados para exportar!")
            return
        
        arquivo = filedialog.asksaveasfilename(
            title="Salvar resultados em Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if arquivo:
            try:
                # Verificar se openpyxl está disponível
                try:
                    import openpyxl
                except ImportError:
                    messagebox.showerror(
                        "Erro", 
                        "Para exportar para Excel, instale a biblioteca openpyxl:\npip install openpyxl"
                    )
                    return
                
                # Criar um escritor Excel com formatação
                with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
                    # Exportar dados principais
                    self.df_resultado.to_excel(writer, sheet_name='Conciliação', index=False)
                    
                    # Obter a planilha para formatação
                    worksheet = writer.sheets['Conciliação']
                    
                    # Ajustar largura das colunas
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    # Adicionar estatísticas em uma nova planilha
                    self._adicionar_estatisticas_excel(writer)
                
                self.adicionar_log(f"📊 Resultados exportados para Excel: {arquivo}")
                messagebox.showinfo("Sucesso", "Resultados exportados em Excel com sucesso!")
                
            except Exception as e:
                self.adicionar_log(f"❌ Erro ao exportar Excel: {e}")
                messagebox.showerror("Erro", f"Erro ao exportar para Excel: {e}")

    def _adicionar_estatisticas_excel(self, writer):
        """Adiciona uma planilha com estatísticas no arquivo Excel."""
        try:
            # Calcular estatísticas
            total_transacoes = len(self.df_resultado)
            categorias_unicas = self.df_resultado['categoria_sugerida'].nunique()
            confianca_media = self.df_resultado['confianca'].mean()
            baixa_confianca = len(self.df_resultado[self.df_resultado['confianca'] < 0.7])
            
            # Distribuição por categoria
            distribuicao = self.df_resultado['categoria_sugerida'].value_counts()
            
            # Criar DataFrame de estatísticas
            estatisticas_data = {
                'Métrica': [
                    'Total de Transações',
                    'Categorias Identificadas',
                    'Confiança Média',
                    'Transações com Baixa Confiança (<70%)',
                    'Data do Processamento'
                ],
                'Valor': [
                    total_transacoes,
                    categorias_unicas,
                    f"{confianca_media:.1%}",
                    baixa_confianca,
                    datetime.now().strftime('%d/%m/%Y %H:%M')
                ]
            }
            
            df_estatisticas = pd.DataFrame(estatisticas_data)
            
            # Criar DataFrame de distribuição
            df_distribuicao = pd.DataFrame({
                'Categoria': distribuicao.index,
                'Quantidade': distribuicao.values,
                'Percentual': [f"{(count/total_transacoes)*100:.1f}%" for count in distribuicao.values]
            })
            
            # Exportar estatísticas
            df_estatisticas.to_excel(writer, sheet_name='Estatísticas', index=False, startrow=0)
            df_distribuicao.to_excel(writer, sheet_name='Estatísticas', index=False, startrow=len(df_estatisticas) + 3)
            
            # Adicionar cabeçalhos
            worksheet_stats = writer.sheets['Estatísticas']
            worksheet_stats.cell(row=len(df_estatisticas) + 3, column=1, value="Distribuição por Categoria:")
            
        except Exception as e:
            self.adicionar_log(f"⚠️ Erro ao adicionar estatísticas: {e}")
    
    def executar(self):
        """Executa a interface gráfica."""
        self.root.mainloop()

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main():
    """Função principal que inicia a interface gráfica."""
    try:
        app = ConciliadorGUI()
        app.executar()
    except Exception as e:
        messagebox.showerror("Erro Fatal", f"Erro ao iniciar aplicação: {e}")

if __name__ == "__main__":
    main()
