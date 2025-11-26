
import base64
import math
import os
import sqlite3
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# Caminhos e constantes
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "rdv.db"
LOGO_PATH = BASE_DIR / "logo-jr.png"
RESAMPLE_FILTER = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS

# A4 em paisagem @300dpi
DPI = 300
MM_PER_INCH = 25.4
A4_WIDTH_MM = 297
A4_HEIGHT_MM = 210
A4_WIDTH_PX = int(A4_WIDTH_MM / MM_PER_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_MM / MM_PER_INCH * DPI)

# Config Neon/Postgres - sempre usa este banco (pode sobrescrever via NEON_DATABASE_URL)
DEFAULT_NEON_URL = "postgresql://neondb_owner:npg_M4mWIzXsPa9Y@ep-dark-fire-ac4gzhg1-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
neon_URL = os.getenv("NEON_DATABASE_URL", DEFAULT_NEON_URL)
neon_HOST = os.getenv("NEON_HOST")
neon_USER = os.getenv("NEON_USER")
neon_PASSWORD = os.getenv("NEON_PASSWORD")
neon_DB = os.getenv("NEON_DB")

# Fallback vazio caso Neon não esteja configurado
COLABORADORES_FALLBACK: list[dict] = []

TIPOS_COLABORADOR = ["MOTORISTA", "AJUDANTE"]

st.set_page_config(page_title="RDV JR", layout="wide")

# -------------------- SQLite RDV --------------------


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = 1;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rdv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                colaborador_nome TEXT NOT NULL,
                tipo TEXT NOT NULL CHECK(tipo IN ('MOTORISTA','AJUDANTE')),
                data_inicial TEXT NOT NULL,
                data_final TEXT NOT NULL,
                adiantamento INTEGER NOT NULL,
                valor_adiantamento REAL NOT NULL,
                total_quinzena REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rdv_linhas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rdv_id INTEGER NOT NULL,
                data TEXT NOT NULL,
                cidade TEXT,
                hotel TEXT,
                valor_hotel REAL,
                diaria_viagem REAL,
                ticket_alimentacao REAL,
                FOREIGN KEY(rdv_id) REFERENCES rdv(id)
            )
            """
        )

def insert_rdv(
    colaborador_nome: str,
    tipo: str,
    data_inicial: date,
    data_final: date,
    adiantamento: bool,
    valor_adiantamento: float,
    total_quinzena: float,
    linhas: list[dict],
) -> None:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rdv (
                colaborador_nome, tipo, data_inicial, data_final,
                adiantamento, valor_adiantamento, total_quinzena
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                colaborador_nome.strip(),
                tipo,
                data_inicial.isoformat(),
                data_final.isoformat(),
                int(adiantamento),
                valor_adiantamento,
                total_quinzena,
            ),
        )
        rdv_id = cur.lastrowid
        valores = []
        for linha in linhas:
            valores.append(
                (
                    rdv_id,
                    linha["DATA"],
                    linha.get("CIDADE") or "",
                    linha.get("HOTEL"),
                    linha.get("VALOR_HOTEL"),
                    linha.get("DIARIA_EM_VIAGEM") or 0,
                    linha.get("TICKET_ALIMENTACAO") or 0,
                )
            )
        cur.executemany(
            """
            INSERT INTO rdv_linhas (
                rdv_id, data, cidade, hotel, valor_hotel, diaria_viagem, ticket_alimentacao
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            valores,
        )
        conn.commit()


def get_rdvs() -> list[dict]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, colaborador_nome, tipo, data_inicial, data_final,
                   adiantamento, valor_adiantamento, total_quinzena
            FROM rdv
            ORDER BY id DESC
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "nome": r[1],
            "tipo": r[2],
            "data_inicial": r[3],
            "data_final": r[4],
            "adiantamento": bool(r[5]),
            "valor_adiantamento": r[6],
            "total_quinzena": r[7],
        }
        for r in rows
    ]


def get_rdv_linhas(rdv_id: int) -> list[dict]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT data, cidade, hotel, valor_hotel, diaria_viagem, ticket_alimentacao
            FROM rdv_linhas
            WHERE rdv_id = ?
            ORDER BY data
            """,
            (rdv_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "DATA": r[0],
            "CIDADE": r[1],
            "HOTEL": r[2],
            "VALOR_HOTEL": r[3],
            "DIARIA_EM_VIAGEM": r[4],
            "TICKET_ALIMENTACAO": r[5],
        }
        for r in rows
    ]


# -------------------- neon colaboradores --------------------


def neon_available() -> bool:
    return bool(neon_URL or (neon_HOST and neon_USER and neon_PASSWORD and neon_DB))


def get_neon_connection():
    if neon_URL:
        return psycopg2.connect(neon_URL, sslmode="require")
    if neon_HOST and neon_USER and neon_PASSWORD and neon_DB:
        return psycopg2.connect(
            host=neon_HOST,
            user=neon_USER,
            password=neon_PASSWORD,
            dbname=neon_DB,
            sslmode="require",
        )
    return None


def init_neon_db() -> None:
    conn = get_neon_connection()
    if not conn:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS colaboradores (
                    id SERIAL PRIMARY KEY,
                    nome TEXT NOT NULL,
                    tipo TEXT NOT NULL CHECK (tipo IN ('MOTORISTA','AJUDANTE'))
                )
                """
            )
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_colaboradores_nome_tipo ON colaboradores(nome, tipo)")
    conn.close()


def neon_get_colaboradores(tipo: Optional[str] = None) -> list[dict]:
    conn = get_neon_connection()
    if not conn:
        return []
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if tipo:
                cur.execute(
                    "SELECT id, nome, tipo FROM colaboradores WHERE tipo = %s ORDER BY nome",
                    (tipo,),
                )
            else:
                cur.execute("SELECT id, nome, tipo FROM colaboradores ORDER BY nome")
            rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def neon_insert_colaborador(nome: str, tipo: str) -> None:
    conn = get_neon_connection()
    if not conn:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO colaboradores (nome, tipo) VALUES (%s, %s)", (nome.strip(), tipo))
    conn.close()


def neon_update_colaborador(colab_id: int, nome: str, tipo: str) -> None:
    conn = get_neon_connection()
    if not conn:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE colaboradores SET nome = %s, tipo = %s WHERE id = %s", (nome.strip(), tipo, colab_id))
    conn.close()


def neon_delete_colaborador(colab_id: int) -> None:
    conn = get_neon_connection()
    if not conn:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM colaboradores WHERE id = %s", (colab_id,))
    conn.close()


def listar_colaboradores_por_tipo(tipo: str) -> list[dict]:
    if neon_available():
        colabs = neon_get_colaboradores(tipo)
        if colabs:
            return colabs
    return [c for c in COLABORADORES_FALLBACK if c["tipo"] == tipo]


def get_all_colaboradores() -> list[dict]:
    if neon_available():
        colabs = neon_get_colaboradores()
        if colabs:
            return colabs
    return COLABORADORES_FALLBACK

# -------------------- Utilidades --------------------


def dates_between(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def try_parse_date(value: date | str) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(value[:10], fmt).date()
            except ValueError:
                continue
    return None


def format_date_br(dt: date | str) -> str:
    parsed = try_parse_date(dt)
    if not parsed:
        return str(dt)
    return parsed.strftime("%d/%m/%Y")


def parse_date_br(value: str) -> date:
    return datetime.strptime(value, "%d/%m/%Y").date()


def format_currency(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0
    if math.isnan(value) or math.isinf(value):
        value = 0
    value = round(value, 2)
    inteiro = int(value)
    cents = int(round((value - inteiro) * 100))
    inteiro_str = f"{inteiro:,}".replace(",", ".")
    return f"R$ {inteiro_str},{cents:02d}"


def to_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        win_font = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
        if win_font.exists():
            return ImageFont.truetype(str(win_font), size)
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def measure_text_width(draw, text, font) -> float:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def draw_text_centered(draw, text, x_center, y, font, fill="black") -> None:
    width = measure_text_width(draw, text, font)
    draw.text((x_center - width / 2, y), text, font=font, fill=fill)


def draw_wrapped_text(draw, text, x, y, font, max_width, line_height: float) -> None:
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if measure_text_width(draw, candidate, font) <= max_width:
            line = candidate
        else:
            if line:
                draw.text((x, y), line, font=font, fill="black")
                y += line_height
            line = word
    if line:
        draw.text((x, y), line, font=font, fill="black")

# -------------------- Geracao da imagem --------------------


def generate_image(
    colaborador_nome: str,
    tipo: str,
    data_inicial: date,
    data_final: date,
    adiantamento: bool,
    valor_adiantamento: float,
    linhas: list[dict],
    mostrar_valores: bool = False,
) -> BytesIO:
    width, height = A4_WIDTH_PX, A4_HEIGHT_PX
    margin_x = int(width * 0.03)
    margin_top = int(height * 0.06)
    bottom_margin = int(height * 0.02)
    reserved_footer = int(height * 0.2)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(int(height * 0.023), bold=True)
    header_font = load_font(int(height * 0.0115), bold=True)
    regular_font = load_font(int(height * 0.0105))
    small_font = load_font(int(height * 0.009))

    y_cursor = margin_top
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        ratio = min((width * 0.035) / logo.width, (height * 0.035) / logo.height, 1.0)
        logo_size = (int(logo.width * ratio), int(logo.height * ratio))
        logo = logo.resize(logo_size, resample=RESAMPLE_FILTER)
        logo_y = margin_top - int(height * 0.015)
        img.paste(logo, (margin_x, logo_y), logo)
        y_cursor = max(y_cursor, logo_y + logo_size[1] + int(height * 0.01))

    title = (
        "RELATORIO DE DESPESAS DE VIAGEM - RDV - MOTORISTA"
        if tipo == "MOTORISTA"
        else "RELATORIO DE DESPESAS DE VIAGEM - RDV - AJUDANTE DE MOTORISTA"
    )
    draw_text_centered(draw, title, width / 2, y_cursor - int(height * 0.03), title_font)

    draw.text((margin_x, y_cursor), f"NOME: {colaborador_nome}", font=regular_font, fill="black")
    draw.text(
        (width * 0.5, y_cursor),
        f"DATA QUINZENA (INICIO E FINAL): {format_date_br(data_inicial)} a {format_date_br(data_final)}",
        font=regular_font,
        fill="black",
    )
    y_cursor += int(height * 0.022)
    draw.text(
        (margin_x, y_cursor),
        "HOUVE adiantamento DE DIARIA? (   ) nao (   ) SIM",
        font=regular_font,
        fill="black",
    )
    draw.text((width * 0.6, y_cursor), "NO VALOR DE R$ _____________________", font=regular_font, fill="black")
    y_cursor += int(height * 0.024)

    table_left = margin_x
    table_right = width - margin_x
    table_width = table_right - table_left
    total_rows = max(len(linhas), 1)

    available_height = height - y_cursor - bottom_margin - reserved_footer
    row_height = max(int(height * 0.017), int(available_height / (total_rows + 1)))
    row_height = min(row_height, int(height * 0.028))
    table_top = y_cursor

    if tipo == "MOTORISTA":
        columns = [
            ("DATA", "DATA", 0.16),
            ("CIDADE", "CIDADE", 0.32),
            ("DIARIA EM VIAGEM", "DIARIA_EM_VIAGEM", 0.26),
            ("TICKET ALIMENTACAO", "TICKET_ALIMENTACAO", 0.26),
        ]
    else:
        columns = [
            ("DATA", "DATA", 0.14),
            ("CIDADE", "CIDADE", 0.2),
            ("HOTEL", "HOTEL", 0.2),
            ("VALOR HOTEL", "VALOR_HOTEL", 0.12),
            ("DIARIA EM VIAGEM", "DIARIA_EM_VIAGEM", 0.17),
            ("TICKET ALIMENTACAO", "TICKET_ALIMENTACAO", 0.17),
        ]

    col_positions: list[tuple[str, str, int, int]] = []
    cursor_x = table_left
    for header, key, pct in columns:
        next_x = cursor_x + int(table_width * pct)
        col_positions.append((header, key, cursor_x, next_x))
        cursor_x = next_x
    if col_positions:
        last = col_positions[-1]
        col_positions[-1] = (last[0], last[1], last[2], table_right)

    table_bottom = table_top + row_height * (total_rows + 1)
    draw.rectangle((table_left, table_top, table_right, table_bottom), outline="black", width=2)
    draw.line((table_left, table_top + row_height, table_right, table_top + row_height), fill="black", width=2)
    for header, _, left, right in col_positions:
        draw.line((left, table_top, left, table_bottom), fill="black", width=2)
        draw.text((left + int(width * 0.0025), table_top + int(row_height * 0.22)), header, font=header_font, fill="black")
    draw.line((table_right, table_top, table_right, table_bottom), fill="black", width=2)

    for idx, linha in enumerate(linhas):
        y = table_top + row_height * (idx + 1)
        draw.line((table_left, y + row_height, table_right, y + row_height), fill="black", width=1)
        is_domingo = False
        for _, key, left, _ in col_positions:
            value = linha.get(key, "")
            text = ""
            if key == "DATA" and value:
                parsed = try_parse_date(value)
                if parsed and parsed.weekday() == 6:
                    text = "DOMINGO"
                    is_domingo = True
                elif parsed:
                    text = parsed.strftime("%d/%m/%Y")
                else:
                    text = str(value)
            elif key in ("DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO", "VALOR_HOTEL"):
                num = to_float(value)
                if mostrar_valores and num != 0:
                    text = format_currency(num)
            else:
                text = "" if value in (None, "") else str(value)
            draw.text((left + int(width * 0.0025), y + int(row_height * 0.18)), text, font=regular_font, fill="black")
        if is_domingo:
            draw.line((table_left, y + row_height / 2, table_right, y + row_height / 2), fill="black", width=1)
    for _, _, _, right in col_positions:
        draw.line((right, table_top, right, table_bottom), fill="black", width=2)

    total_y = table_bottom + int(height * 0.01)
    draw.text((table_left, total_y), "TOTAL DA QUINZENA EM R$ -----> R$ __________________", font=header_font, fill="black")

    loc_y = total_y + int(height * 0.02)
    draw.text((margin_x, loc_y), "LOCAL/DATA:", font=regular_font, fill="black")
    draw.text((margin_x + int(width * 0.12), loc_y), "______________________________", font=regular_font, fill="black")
    date_y = loc_y + int(height * 0.017)
    draw.text((margin_x, date_y), ", ____________  de  ____________  de  ____________", font=regular_font, fill="black")

    sign_label_y = date_y + int(height * 0.06)
    line_y = sign_label_y + int(height * 0.055)
    signature_width = (width - 2 * margin_x) / 3
    max_line = signature_width - int(width * 0.02)
    labels = ["ASSINATURA COLABORADOR", "ANALISTA FROTA", "GESTOR FROTA"]
    for idx, label in enumerate(labels):
        x = margin_x + idx * signature_width
        draw.text((x + 10, sign_label_y), label, font=regular_font, fill="black")
        draw.line((x + 5, line_y, x + 5 + max_line, line_y), fill="black", width=2)

    obs_y = line_y + int(height * 0.035)
    obs_text = (
        "OBSERVACAO: Nos termos da Convencao Coletiva a diaria de viagem e destinada apenas ao colaborador "
        "que exercer atividade fora da base consideranao cada periodo modular de 24 horas, "
        "o recebimenao da diaria exclui-se o pagamenao da ajuda de alimentacao (Ticket)."
    )
    obs_line_height = getattr(small_font, "size", 18) + 4
    draw_wrapped_text(draw, obs_text, margin_x, obs_y, small_font, width - 2 * margin_x, obs_line_height)

    buffer = BytesIO()
    img.save(buffer, format="PNG", dpi=(DPI, DPI))
    buffer.seek(0)
    return buffer

# -------------------- Print helpers --------------------


def open_print_window(image_data: bytes) -> None:
    _open_print_window([image_data])


def open_print_window_batch(images: list[bytes]) -> None:
    if images:
        _open_print_window(images)


def _open_print_window(images: list[bytes]) -> None:
    encoded_images = [base64.b64encode(img).decode("ascii") for img in images]
    wrappers = "".join(
        f'<div class="rdv-wrapper"><img src="data:image/png;base64,{img}" /></div>' for img in encoded_images
    )
    page_break_style = "page-break-after: always;" if len(encoded_images) > 1 else "page-break-after: auto;"
    components.html(
        f"""
        <script>
            const w = window.open('', '_blank');
            if (w) {{
                w.document.write(`
                    <html>
                        <head>
                            <meta charset='utf-8' />
                            <style>
                                @page {{
                                    size: A4 landscape;
                                    margin: 0;
                                }}
                                html, body {{
                                    margin: 0;
                                    padding: 0;
                                    width: 100%;
                                    height: 100%;
                                }}
                                body {{
                                    background: #fff;
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                }}
                                .rdv-wrapper {{
                                    width: 100%;
                                    {page_break_style}
                                    display: flex;
                                    justify-content: center;
                                    align-items: center;
                                }}
                                .rdv-wrapper img {{
                                    width: 100%;
                                    height: auto;
                                    display: block;
                                }}
                            </style>
                        </head>
                        <body>{wrappers}</body>
                    </html>
                `);
                w.document.close();
                w.focus();
                w.print();
            }} else {{
                alert('Habilite pop-ups para imprimir o RDV.');
            }}
        </script>
        """,
        height=0,
    )


# -------------------- Construcao de tabela --------------------


def sum_total(rows: list[dict]) -> float:
    total = 0.0
    for row in rows:
        for key in ("DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO", "VALOR_HOTEL"):
            total += to_float(row.get(key, 0))
    return total


def build_table_dataframe(tipo: str, start_date: date, end_date: date) -> pd.DataFrame:
    cols = ["DATA", "CIDADE"]
    if tipo == "AJUDANTE":
        cols += ["HOTEL", "VALOR_HOTEL"]
    cols += ["DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO"]
    rows = []
    for current in dates_between(start_date, end_date):
        rows.append(
            {
                "DATA": format_date_br(current),
                "CIDADE": "",
                "HOTEL": "" if tipo == "AJUDANTE" else None,
                "VALOR_HOTEL": 0.0 if tipo == "AJUDANTE" else None,
                "DIARIA_EM_VIAGEM": 0.0,
                "TICKET_ALIMENTACAO": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def ensure_session_state() -> None:
    st.session_state.setdefault("rdv_table", None)
    st.session_state.setdefault("rdv_meta", {})
    st.session_state.setdefault("generated_image_data", None)
    st.session_state.setdefault("generated_image_name", "")
    st.session_state.setdefault("generated_image_page", "")
    st.session_state.setdefault("batch_previews", {"tipo": "", "previews": []})

# -------------------- UI helpers --------------------


def get_default_quinzena() -> tuple[date, date]:
    duration = timedelta(days=12)  # 13 dias incluindo início/fim
    today = date.today()

    def next_monday_on_or_after(d: date) -> date:
        while d.weekday() != 0:  # 0 = segunda
            d += timedelta(days=1)
        return d

    # Ponto de partida: próxima segunda-feira a partir de hoje
    candidate = next_monday_on_or_after(today)

    # Se houver RDV salvo e o último fim for maior/igual ao candidato,
    # pulamos para a próxima segunda após o último fim para evitar sobreposição.
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT data_final FROM rdv ORDER BY id DESC LIMIT 1")
        last = cur.fetchone()
    if last:
        try:
            last_final = datetime.fromisoformat(last[0]).date()
            if last_final >= candidate:
                candidate = next_monday_on_or_after(last_final + timedelta(days=1))
        except Exception:
            pass

    end = candidate + duration
    return candidate, end


def render_generated_image(page: str) -> None:
    data = st.session_state.get("generated_image_data")
    where = st.session_state.get("generated_image_page")
    if data and where == page:
        st.image(data, use_container_width=True)
        st.download_button(
            "Baixar RDV (PNG)",
            data=data,
            file_name=st.session_state.get("generated_image_name", "rdv.png"),
            mime="image/png",
        )
        if st.button("Imprimir RDV gerado", key=f"print_{page}"):
            open_print_window(data)


# -------------------- Paginas --------------------


def pagina_colaboradores() -> None:
    st.header("Colaboradores (Neon)")
    neon_ok = neon_available()
    if not neon_ok:
        st.info("Neon nao configurado. Mostrando lista somente para referencia (nao editavel).")
    col1, col2 = st.columns([3, 1])
    with col1:
        nome_novo = st.text_input("Nome do colaborador")
    with col2:
        tipo_novo = st.selectbox("Tipo", TIPOS_COLABORADOR, key="tipo_novo")
    if st.button("Salvar colaborador", disabled=not neon_ok):
        if not nome_novo.strip():
            st.error("Informe o nome.")
        else:
            neon_insert_colaborador(nome_novo, tipo_novo)
            st.success("Colaborador salvo.")
            st.rerun()

    colabs = get_all_colaboradores()
    if not colabs:
        st.info("Nenhum colaborador cadastrado.")
        return

    nomes_map = {f"{c.get('nome')} ({c.get('tipo')})": c for c in colabs}
    selecionado = st.selectbox("Editar/Excluir", list(nomes_map.keys()))
    csel = nomes_map[selecionado]
    novo_nome = st.text_input("Nome", value=csel.get("nome", ""), key="edit_nome")
    novo_tipo = st.selectbox("Tipo", TIPOS_COLABORADOR, index=TIPOS_COLABORADOR.index(csel.get("tipo", "MOTORISTA")), key="edit_tipo")
    cols_btn = st.columns(2)
    with cols_btn[0]:
            if st.button("Atualizar", disabled=not neon_ok):
                neon_update_colaborador(int(csel.get("id", 0)), novo_nome, novo_tipo)
                st.success("Atualizado.")
                st.rerun()
    with cols_btn[1]:
        if st.button("Excluir", disabled=not neon_ok):
            neon_delete_colaborador(int(csel.get("id", 0)))
            st.warning("Excluido.")
            st.rerun()

    st.subheader("Lista")
    st.dataframe(pd.DataFrame(colabs), use_container_width=True)


def pagina_novo_rdv() -> None:
    st.header("Novo RDV")
    tipo = st.selectbox("Tipo de colaborador", TIPOS_COLABORADOR, format_func=lambda t: "Motorista" if t == "MOTORISTA" else "Ajudante")
    colaboradores = listar_colaboradores_por_tipo(tipo)
    if not colaboradores:
        st.warning("Nenhum colaborador para este tipo.")
        return
    modo_all_label = f"Todos os {tipo.lower()}s"
    modo = st.radio("Modo de Geracao", ["Individual", modo_all_label], horizontal=True)
    data_ini, data_fim = get_default_quinzena()
    data_inicial = st.date_input("Data inicial", value=data_ini)
    data_final = st.date_input("Data final", value=data_fim)
    if data_final < data_inicial:
        st.error("A data final deve ser igual ou posterior a inicial.")
        return
    adiantamento_flag = st.checkbox("Houve adiantamento de diaria?")
    valor_adiantamento = 0.0
    if adiantamento_flag:
        valor_adiantamento = st.number_input("Valor do adiantamento (R$)", min_value=0.0, value=0.0, step=10.0, format="%f")

    if modo == "Individual":
        nomes = [c["nome"] for c in colaboradores]
        nome_escolhido = st.selectbox("Colaborador", nomes)
        colaborador = next(c for c in colaboradores if c["nome"] == nome_escolhido)
        if st.button("Gerar tabela da quinzena"):
            st.session_state["rdv_table"] = build_table_dataframe(tipo, data_inicial, data_final)
            st.session_state["rdv_meta"] = {
                "tipo": tipo,
                "colaborador_nome": colaborador["nome"],
                "data_inicial": data_inicial,
                "data_final": data_final,
                "adiantamento": adiantamento_flag,
                "valor_adiantamento": valor_adiantamento,
            }
        df = st.session_state.get("rdv_table")
        if df is not None:
            column_config = {"DATA": st.column_config.Column("Data", disabled=True)}
            st.session_state["rdv_table"] = st.data_editor(df, column_config=column_config, use_container_width=True)
            df = st.session_state["rdv_table"]
            total = sum_total(df.to_dict("records"))
            st.markdown(f"**TOTAL DA QUINZENA EM R$:** {format_currency(total)}")
            if st.button("Salvar RDV"):
                if df.empty:
                    st.error("Tabela vazia.")
                else:
                    linhas = []
                    for _, row in df.iterrows():
                        linhas.append(
                            {
                                "DATA": parse_date_br(row["DATA"]).isoformat(),
                                "CIDADE": row.get("CIDADE", ""),
                                "HOTEL": row.get("HOTEL") if tipo == "AJUDANTE" else None,
                                "VALOR_HOTEL": to_float(row.get("VALOR_HOTEL")) if tipo == "AJUDANTE" else None,
                                "DIARIA_EM_VIAGEM": to_float(row.get("DIARIA_EM_VIAGEM")),
                                "TICKET_ALIMENTACAO": to_float(row.get("TICKET_ALIMENTACAO")),
                            }
                        )
                    insert_rdv(
                        colaborador["nome"],
                        tipo,
                        data_inicial,
                        data_final,
                        adiantamento_flag,
                        valor_adiantamento if adiantamento_flag else 0.0,
                        total,
                        linhas,
                    )
                    st.success("RDV salvo.")
                    st.session_state["rdv_table"] = None
                    st.session_state["rdv_meta"] = {}
            if st.button("Gerar imagem do RDV (PNG)"):
                meta = st.session_state.get("rdv_meta", {})
                if not meta:
                    st.error("Gere a tabela antes.")
                else:
                    img_buffer = generate_image(
                        colaborador_nome=colaborador["nome"],
                        tipo=tipo,
                        data_inicial=data_inicial,
                        data_final=data_final,
                        adiantamento=adiantamento_flag,
                        valor_adiantamento=valor_adiantamento if adiantamento_flag else 0.0,
                        linhas=df.to_dict("records"),
                        mostrar_valores=True,
                    )
                    st.session_state["generated_image_data"] = img_buffer.getvalue()
                    st.session_state["generated_image_name"] = (
                        f"RDV_{colaborador['nome'].replace(' ', '_')}_{tipo}_"
                        f"{data_inicial.strftime('%Y%m%d')}-{data_final.strftime('%Y%m%d')}.png"
                    )
                    st.session_state["generated_image_page"] = "Novo RDV"
        render_generated_image("Novo RDV")
    else:
        if st.button(f"Gerar RDVs para todos os {tipo.lower()}s"):
            template = build_table_dataframe(tipo, data_inicial, data_final).to_dict("records")
            previews = []
            for colab in colaboradores:
                rows_copy = [r.copy() for r in template]
                img_buffer = generate_image(
                    colaborador_nome=colab["nome"],
                    tipo=tipo,
                    data_inicial=data_inicial,
                    data_final=data_final,
                    adiantamento=adiantamento_flag,
                    valor_adiantamento=valor_adiantamento if adiantamento_flag else 0.0,
                    linhas=rows_copy,
                )
                previews.append(
                    {
                        "nome": colab["nome"],
                        "image": img_buffer.getvalue(),
                        "file_name": f"RDV_{colab['nome'].replace(' ', '_')}_{tipo}_{data_inicial.strftime('%Y%m%d')}-{data_final.strftime('%Y%m%d')}.png",
                    }
                )
            st.session_state["batch_previews"] = {"tipo": tipo, "previews": previews}
            st.success(f"Gerados {len(previews)} RDVs.")
        batch = st.session_state.get("batch_previews", {})
        if batch.get("tipo") == tipo:
            for idx, preview in enumerate(batch["previews"]):
                st.subheader(preview["nome"])
                st.image(preview["image"], use_container_width=True)
                st.download_button(
                    "Baixar RDV (PNG)",
                    data=preview["image"],
                    file_name=preview["file_name"],
                    mime="image/png",
                    key=f"dl_{tipo}_{idx}",
                )
                if st.button("Imprimir RDV", key=f"pr_{tipo}_{idx}"):
                    open_print_window(preview["image"])
            if batch.get("previews"):
                if st.button(f"Imprimir todos os {tipo.lower()}s", key=f"pr_all_{tipo}"):
                    open_print_window_batch([p["image"] for p in batch["previews"]])


def pagina_relatorios() -> None:
    st.header("Relatorios salvos")
    rdvs = get_rdvs()
    if not rdvs:
        st.info("Nenhum RDV salvo.")
        return
    labels = {
        f"{r['nome']} | {r['tipo']} | {format_date_br(r['data_inicial'])} a {format_date_br(r['data_final'])} (ID: {r['id']})": r
        for r in rdvs
    }
    selecionado = st.selectbox("RDV", options=list(labels.keys()))
    rdv_data = labels[selecionado]
    linhas = get_rdv_linhas(rdv_data["id"])
    st.markdown(
        f"**Colaborador:** {rdv_data['nome']} | **Tipo:** {rdv_data['tipo']} | "
        f"**Quinzena:** {format_date_br(rdv_data['data_inicial'])} a {format_date_br(rdv_data['data_final'])}"
    )
    st.markdown(
        f"**Total da quinzena:** {format_currency(rdv_data['total_quinzena'])} | "
        f"**adiantamento:** {'Sim' if rdv_data['adiantamento'] else 'nao'} "
        f"{format_currency(rdv_data['valor_adiantamento']) if rdv_data['adiantamento'] else ''}"
    )
    df = pd.DataFrame(linhas)
    if rdv_data["tipo"] == "MOTORISTA":
        st.dataframe(df[["DATA", "CIDADE", "DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO"]])
    else:
        st.dataframe(df[["DATA", "CIDADE", "HOTEL", "VALOR_HOTEL", "DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO"]])
    if st.button("Gerar imagem do RDV (PNG)"):
        img_buffer = generate_image(
            colaborador_nome=rdv_data["nome"],
            tipo=rdv_data["tipo"],
            data_inicial=datetime.fromisoformat(rdv_data["data_inicial"]).date(),
            data_final=datetime.fromisoformat(rdv_data["data_final"]).date(),
            adiantamento=rdv_data["adiantamento"],
            valor_adiantamento=rdv_data["valor_adiantamento"],
            linhas=linhas,
            mostrar_valores=True,
        )
        st.session_state["generated_image_data"] = img_buffer.getvalue()
        st.session_state["generated_image_name"] = (
            f"RDV_{rdv_data['nome'].replace(' ', '_')}_{rdv_data['tipo']}_"
            f"{datetime.fromisoformat(rdv_data['data_inicial']).strftime('%Y%m%d')}-"
            f"{datetime.fromisoformat(rdv_data['data_final']).strftime('%Y%m%d')}.png"
        )
        st.session_state["generated_image_page"] = "Relatorios salvos"
    render_generated_image("Relatorios salvos")


def main() -> None:
    ensure_session_state()
    init_db()
    init_neon_db()

    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(LOGO_PATH, width=80)
        else:
            st.markdown("JR")
    with title_col:
        st.title("Relatorio de Despesas de Viagem - RDV JR")

    page = st.sidebar.selectbox("Menu", ["Colaboradores", "Novo RDV", "Relatorios salvos"])
    if page == "Colaboradores":
        pagina_colaboradores()
    elif page == "Novo RDV":
        pagina_novo_rdv()
    else:
        pagina_relatorios()


if __name__ == "__main__":
    main()













