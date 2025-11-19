# RDV JR – Relatório de Despesas de Viagem

Aplicativo local em Streamlit que controla colaboradores (motoristas e ajudantes), gera relatórios quinzenais e cria o PNG oficial do RDV pronto para imprimir.

## Rodando localmente

1. Instale dependências: `pip install -r requirements.txt`
2. Execute com: `streamlit run app.py`
3. [Opcional] Garanta que o `rdv.db` seja criado automaticamente automaticamente ao abrir o app.

## Versionamento e deploy

- O projeto já está preparado para GitHub (adicione os arquivos e faça `git push` para `https://github.com/jhonnyoraculos/rdv-jr.git`).
- Para implantar na Streamlit Community Cloud, conecte o repositório, use o `main file` `app.py` e o `requirements.txt` será utilizado automaticamente.

## Estrutura

- `app.py`: aplicação Streamlit principal com persistência SQLite, cadastro de colaboradores, geração dos RDVs e exportação de PNG/impressão.
- `requirements.txt`: dependências mínimas.
- `.gitignore`: evita comitar arquivos temporários e o banco local (`rdv.db`).
