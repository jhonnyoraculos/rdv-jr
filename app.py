import math
import sqlite3
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import base64
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "rdv.db"
LOGO_PATH = BASE_DIR / "logo-jr.png"
LOGO_MAX_WIDTH = 200
LOGO_MAX_HEIGHT = 160
RESAMPLE_FILTER = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
)

# ==========================
# CADASTRO FIXO DE COLABORADORES
# ==========================
COLABORADORES = [
    # Motoristas
    {"nome": "ALDEMIR LUIZ DA SILVA", "tipo": "MOTORISTA"},
    {"nome": "ANDRE LUIZ", "tipo": "MOTORISTA"},
    {"nome": "CELSO ANTONIO CAETANO", "tipo": "MOTORISTA"},
    {"nome": "CRISTIANO CLEMENTINO OLIVEIRA", "tipo": "MOTORISTA"},
    {"nome": "DIEGO GERALDO BAZILIO", "tipo": "MOTORISTA"},
    {"nome": "DOUGLAS ALBERTINO GREGORIO", "tipo": "MOTORISTA"},
    {"nome": "DOUGLAS RODRIGUES DE OLIVEIRA", "tipo": "MOTORISTA"},
    {"nome": "FRANCES FRANCO", "tipo": "MOTORISTA"},
    {"nome": "FRANCIS EDER NUNES", "tipo": "MOTORISTA"},
    {"nome": "FREDER HENRIQUE MOREIRA DE CARVALHO", "tipo": "MOTORISTA"},
    {"nome": "GABRIEL FELIPE DE FARIA OLIVEIRA", "tipo": "MOTORISTA"},
    {"nome": "GERALDO FERNANDO DA SILVA", "tipo": "MOTORISTA"},
    {"nome": "GUILHERME FLAVIO DOS SANTOS", "tipo": "MOTORISTA"},
    {"nome": "HIPOCRATES HERSCHEL PINTO", "tipo": "MOTORISTA"},
    {"nome": "IAGO RAIMUNDO DIAS", "tipo": "MOTORISTA"},
    {"nome": "JOSE ARILDO DOMINGOS", "tipo": "MOTORISTA"},
    {"nome": "KAIO FERNANDO", "tipo": "MOTORISTA"},
    {"nome": "LUCAS APARECIDO ROQUE", "tipo": "MOTORISTA"},
    {"nome": "LUCAS SILVA NOGUEIRA", "tipo": "MOTORISTA"},
    {"nome": "MATEUS SEVERINO DE SOUZA", "tipo": "MOTORISTA"},
    {"nome": "MATHEUS RINALDO PEREIRA VAZ", "tipo": "MOTORISTA"},
    {"nome": "PAULO ROGERIO GONÇALVES AZEVEDO", "tipo": "MOTORISTA"},
    {"nome": "PEDRO AMARAL E SILVA", "tipo": "MOTORISTA"},
    {"nome": "RAYNER FRANCIS LOPES DE ALMEIDA", "tipo": "MOTORISTA"},
    {"nome": "REGINALDO MOREIRA LÃO", "tipo": "MOTORISTA"},
    {"nome": "RICARDO DE OLIVEIRA SILVA", "tipo": "MOTORISTA"},
    {"nome": "RONALDO PEREIRA CORDEIRO", "tipo": "MOTORISTA"},
    {"nome": "SIDNEY RAIMUNDO DA SILVA", "tipo": "MOTORISTA"},
    {"nome": "WESLEY ANTONIO SENA DA SILVA", "tipo": "MOTORISTA"},
    {"nome": "WESLEY LUCIO", "tipo": "MOTORISTA"},
    {"nome": "HELIO APARECIDO CANEDO", "tipo": "MOTORISTA"},

    # Ajudantes
    {"nome": "ANDRE LUIS GONÇALVES PEREIRA", "tipo": "AJUDANTE"},
    {"nome": "BRUNO HENRIQUE MENDES", "tipo": "AJUDANTE"},
    {"nome": "CHARLES COSTA SANTOS", "tipo": "AJUDANTE"},
    {"nome": "DEVIS PENA DE OLIVEIRA", "tipo": "AJUDANTE"},
    {"nome": "EDER SILVA", "tipo": "AJUDANTE"},
    {"nome": "EDUARDO ANDRADE SILVA", "tipo": "AJUDANTE"},
    {"nome": "ELDERSON JOSE GOMES", "tipo": "AJUDANTE"},
    {"nome": "EMERSON FELIPE MACHADO", "tipo": "AJUDANTE"},
    {"nome": "ERASMO ROBERTO LOPES GONÇALVES", "tipo": "AJUDANTE"},
    {"nome": "FABRICIO DA SILVA SOUSA", "tipo": "AJUDANTE"},
    {"nome": "HIAGO HENRIQUE LOPES", "tipo": "AJUDANTE"},
    {"nome": "JOAO HELIO SILVA LACERDA", "tipo": "AJUDANTE"},
    {"nome": "KENEDY DEIVISON LOPES PEREIRA", "tipo": "AJUDANTE"},
    {"nome": "LAENDER LOURENÇO DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "LUCAS GABRIEL DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "LUIS EDUARDO CUSTODIO COELHO", "tipo": "AJUDANTE"},
    {"nome": "MARCELO DA CONCEIÇÃO SANTOS", "tipo": "AJUDANTE"},
    {"nome": "MARCOS PAULO MILAGRES DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "MARLON GERALDO ALVES SILVA", "tipo": "AJUDANTE"},
    {"nome": "MAYCOL LUCAS MENDES DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "ORMIR GONÇALVES BORGES", "tipo": "AJUDANTE"},
    {"nome": "PABLO HENRIQUE NOGUEIRA GONTIJO", "tipo": "AJUDANTE"},
    {"nome": "REGINALDO LAURO SANTOS ABREU", "tipo": "AJUDANTE"},
    {"nome": "RICHARD SANTOS LOPES", "tipo": "AJUDANTE"},
    {"nome": "ROBERT JHONATHAN SILVA", "tipo": "AJUDANTE"},
    {"nome": "RUAN CARLOS DIAS FRANCO DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "RYCHARD MARTINS DA SILVA", "tipo": "AJUDANTE"},
    {"nome": "WELLINGTON GUSTAVO SANTOS", "tipo": "AJUDANTE"},
    {"nome": "WEVERSON FERREIRA DOS SANTOS", "tipo": "AJUDANTE"},
]

TIPOS_COLABORADOR = ["MOTORISTA", "AJUDANTE"]


def listar_colaboradores_por_tipo(tipo: str) -> list[dict]:
    """Retorna apenas os colaboradores do tipo informado."""
    return [c for c in COLABORADORES if c["tipo"] == tipo]


st.set_page_config(page_title="RDV JR", layout="wide")


def get_connection() -> sqlite3.Connection:
    """Return a connection to the local SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = 1;")
    return conn


def init_db() -> None:
    """Create RDV tables if they do not already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rdv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                colaborador_nome TEXT NOT NULL,
                tipo TEXT NOT NULL CHECK(tipo IN ('MOTORISTA','AJUDANTE')),
                data_inicial TEXT NOT NULL,
                data_final TEXT NOT NULL,
                adiantamento INTEGER NOT NULL,
                valor_adiantamento REAL NOT NULL,
                total_quinzena REAL NOT NULL,
                colaborador_id INTEGER
            )
            """
        )
        cursor.execute(
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
        cursor.execute("PRAGMA table_info(rdv)")
        rdv_columns = [col[1] for col in cursor.fetchall()]
        if "colaborador_nome" not in rdv_columns:
            cursor.execute("ALTER TABLE rdv ADD COLUMN colaborador_nome TEXT NOT NULL DEFAULT ''")
            rdv_columns.append("colaborador_nome")
        if "colaborador_id" not in rdv_columns:
            cursor.execute("ALTER TABLE rdv ADD COLUMN colaborador_id INTEGER")
            rdv_columns.append("colaborador_id")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        if "colaboradores" in existing_tables:
            cursor.execute(
                """
                UPDATE rdv
                SET colaborador_nome = (
                    SELECT nome FROM colaboradores WHERE colaboradores.id = rdv.colaborador_id
                )
                WHERE colaborador_nome = ''
                """
            )


def get_default_quinzena() -> tuple[date, date]:
    """Return default quinzena anchored on next available Monday."""
    fallback_start = date(2025, 11, 10)
    duration = timedelta(days=12)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data_final FROM rdv ORDER BY id DESC LIMIT 1")
        last = cursor.fetchone()
    if last:
        try:
            last_final = datetime.fromisoformat(last[0]).date()
            candidate = last_final + timedelta(days=1)
        except Exception:
            candidate = fallback_start
    else:
        candidate = fallback_start
    while candidate.weekday() != 0:  # 0 = Monday
        candidate += timedelta(days=1)
    return candidate, candidate + duration


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
    """Persist an RDV header and its lines into the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
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
        rdv_id = cursor.lastrowid
        linha_values = []
        for linha in linhas:
            linha_values.append(
                (
                    rdv_id,
                    linha["DATA"],
                    linha.get("CIDADE") or "",
                    linha.get("HOTEL"),
                    linha.get("VALOR_HOTEL") or None,
                    linha.get("DIARIA_EM_VIAGEM") or 0,
                    linha.get("TICKET_ALIMENTACAO") or 0,
                )
            )
        cursor.executemany(
            """
            INSERT INTO rdv_linhas (
                rdv_id, data, cidade, hotel, valor_hotel, diaria_viagem, ticket_alimentacao
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            linha_values,
        )
        conn.commit()


def get_rdvs() -> list[dict]:
    """Return saved RDVs with collaborator metadata."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                colaborador_nome,
                tipo,
                data_inicial,
                data_final,
                adiantamento,
                valor_adiantamento,
                total_quinzena
            FROM rdv
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()
    return [
        {
            "id": row[0],
            "nome": row[1],
            "tipo": row[2],
            "data_inicial": row[3],
            "data_final": row[4],
            "adiantamento": bool(row[5]),
            "valor_adiantamento": row[6],
            "total_quinzena": row[7],
        }
        for row in rows
    ]


def get_rdv_linhas(rdv_id: int) -> list[dict]:
    """Return the lines that belong to a saved RDV."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT data, cidade, hotel, valor_hotel, diaria_viagem, ticket_alimentacao
            FROM rdv_linhas
            WHERE rdv_id = ?
            ORDER BY data
            """,
            (rdv_id,),
        )
        rows = cursor.fetchall()
    linhas = []
    for row in rows:
        linhas.append(
            {
                "DATA": row[0],
                "CIDADE": row[1],
                "HOTEL": row[2],
                "VALOR_HOTEL": row[3],
                "DIARIA_EM_VIAGEM": row[4],
                "TICKET_ALIMENTACAO": row[5],
            }
        )
    return linhas


def dates_between(start_date: date, end_date: date) -> list[date]:
    """Return a list of dates between start and end inclusive."""
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def try_parse_date(value: date | str) -> date | None:
    """Try common date formats and return a date object."""
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
    """Return date formatted as dd/mm/aaaa."""
    parsed = try_parse_date(dt)
    if not parsed:
        return str(dt)
    return parsed.strftime("%d/%m/%Y")


def parse_date_br(value: str) -> date:
    """Parse a brazilian formatted date string."""
    return datetime.strptime(value, "%d/%m/%Y").date()


def format_currency(value: float) -> str:
    """Format a float as Brazilian currency (R$ 0,00)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0
    if math.isnan(value) or math.isinf(value):
        value = 0
    value = round(value, 2)
    integer_part = int(value)
    cents = int(round((value - integer_part) * 100))
    integer_str = f"{integer_part:,}".replace(",", ".")
    return f"R$ {integer_str},{cents:02d}"


def to_float(value: float | str | None) -> float:
    """Convert different types to a safe float, returning 0 when invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return number


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a font for Pillow, falling back to default if necessary."""
    try:
        system_font = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
        if system_font.exists():
            return ImageFont.truetype(str(system_font), size)
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def measure_text_width(draw, text, font) -> float:
    """Return the text width using available PIL helpers."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        return draw.textsize(text, font=font)[0]


def draw_text_centered(draw, text, x_center, y, font, fill="black") -> None:
    """Helper to draw centered text horizontally."""
    width = measure_text_width(draw, text, font)
    draw.text((x_center - width / 2, y), text, font=font, fill=fill)


def draw_wrapped_text(
    draw,
    text,
    x,
    y,
    font,
    max_width,
    line_height: float,
) -> None:
    """Draw text wrapping automatically to stay within the specified width."""
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
    """Build a PNG image version of the RDV."""
    width, height = 3508, 2480  # A4 at 300dpi: 3508x2480
    margin = 40
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(48, bold=True)
    header_font = load_font(32, bold=True)
    regular_font = load_font(28)
    small_font = load_font(20)

    # Header with logo and title
    y_cursor = margin
    if LOGO_PATH.exists():
        logo = Image.open(LOGO_PATH).convert("RGBA")
        ratio = min(
            LOGO_MAX_WIDTH / logo.width,
            LOGO_MAX_HEIGHT / logo.height,
            1.0,
        )
        logo_size = (int(logo.width * ratio), int(logo.height * ratio))
        logo = logo.resize(logo_size, resample=RESAMPLE_FILTER)
        img.paste(logo, (margin, margin), logo)
    title = (
        "RELATÓRIO DE DESPESAS DE VIAGEM - RDV - MOTORISTA"
        if tipo == "MOTORISTA"
        else "RELATÓRIO DE DESPESAS DE VIAGEM - RDV - AJUDANTE DE MOTORISTA"
    )
    draw_text_centered(draw, title, width / 2, margin, title_font)
    y_cursor += 120

    # Collaborator metadata
    draw.text(
        (margin, y_cursor),
        f"NOME: {colaborador_nome}",
        font=regular_font,
        fill="black",
    )
    draw.text(
        (width * 0.55, y_cursor),
        f"DATA QUINZENA (INICIO E FINAL): {format_date_br(data_inicial)} a {format_date_br(data_final)}",
        font=regular_font,
        fill="black",
    )
    y_cursor += 40
    draw.text(
        (margin, y_cursor),
        "HOUVE ADIANTAMENTO DE DIÁRIA? ( ) NÃO ( ) SIM",
        font=regular_font,
        fill="black",
    )
    valor_text = "NO VALOR DE R$ _____________________"
    draw.text((width * 0.6, y_cursor), valor_text, font=regular_font, fill="black")
    y_cursor += 60

    # Table layout
    table_left = margin
    table_right = width - margin
    table_width = table_right - table_left
    total_rows = max(len(linhas), 1)
    row_height = 60
    total_table_height = row_height * (total_rows + 1)
    table_top = y_cursor + 30
    if tipo == "MOTORISTA":
        columns = [
            ("DATA", "DATA", 0.15),
            ("CIDADE", "CIDADE", 0.35),
            ("DIÁRIA EM VIAGEM", "DIARIA_EM_VIAGEM", 0.25),
            ("TICKET ALIMENTAÇÃO", "TICKET_ALIMENTACAO", 0.25),
        ]
    else:
        columns = [
            ("DATA", "DATA", 0.13),
            ("CIDADE", "CIDADE", 0.22),
            ("HOTEL", "HOTEL", 0.25),
            ("VALOR HOTEL", "VALOR_HOTEL", 0.12),
            ("DIÁRIA EM VIAGEM", "DIARIA_EM_VIAGEM", 0.14),
            ("TICKET ALIMENTAÇÃO", "TICKET_ALIMENTACAO", 0.14),
        ]

    column_positions: list[tuple[str, str, int, int]] = []
    cursor_pos = table_left
    for header, key, pct in columns:
        next_cursor = cursor_pos + int(table_width * pct)
        column_positions.append((header, key, cursor_pos, next_cursor))
        cursor_pos = next_cursor
    if column_positions:
        column_positions[-1] = (
            column_positions[-1][0],
            column_positions[-1][1],
            column_positions[-1][2],
            table_right,
        )

    table_bottom = table_top + row_height * (total_rows + 1)
    # table boundary
    draw.rectangle((table_left, table_top, table_right, table_bottom), outline="black", width=2)

    # Header row
    draw.line((table_left, table_top + row_height, table_right, table_top + row_height), fill="black", width=2)
    for header, _, left, right in column_positions:
        draw.line((left, table_top, left, table_bottom), fill="black", width=2)
        draw.text(
            (left + 5, table_top + 10),
            header,
            font=header_font,
            fill="black",
        )
    draw.line((table_right, table_top, table_right, table_bottom), fill="black", width=2)

    # Rows
    for idx, linha in enumerate(linhas):
        y = table_top + row_height * (idx + 1)
        draw.line((table_left, y + row_height, table_right, y + row_height), fill="black", width=1)
        is_domingo = False
        for header, key, left, right in column_positions:
            value = linha.get(key, "")
            if key == "DATA" and value:
                parsed_date = try_parse_date(value)
                if parsed_date and parsed_date.weekday() == 6:
                    text = "DOMINGO"
                    is_domingo = True
                elif parsed_date:
                    text = parsed_date.strftime("%d/%m/%Y")
                else:
                    text = str(value)
            elif key in ("DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO", "VALOR_HOTEL"):
                numeric_value = to_float(value)
                if mostrar_valores and numeric_value != 0:
                    text = format_currency(numeric_value)
                else:
                    text = ""
            else:
                text = str(value) if value not in (None, "") else ""
            draw.text((left + 5, y + 10), text, font=regular_font, fill="black")
        if is_domingo:
            draw.line(
                (table_left, y + row_height / 2, table_right, y + row_height / 2),
                fill="black",
                width=1,
            )
    for _, _, _, right in column_positions:
        draw.line((right, table_top, right, table_bottom), fill="black", width=2)

    # Totals
    total_text = "TOTAL DA QUINZENA EM R$ -----> R$ _____________________"
    total_y = table_bottom + 20
    draw.text((table_left, total_y), total_text, font=header_font, fill="black")

    # Local/data row
    loc_y = total_y + 50
    draw.text(
        (margin, loc_y),
        "LOCAL/DATA: ________________________________",
        font=regular_font,
        fill="black",
    )
    draw.text(
        (width * 0.65, loc_y),
        "_____/_____/______",
        font=regular_font,
        fill="black",
    )

    # Signatures / footer
    sign_label_y = loc_y + 60
    signature_width = (width - 2 * margin) / 3
    max_line_width = signature_width - 20
    for idx, text in enumerate(
        ["ASSINATURA COLABORADOR", "ANALISTA FROTA", "GESTOR FROTA"]
    ):
        x = margin + idx * signature_width
        draw.text((x + 10, sign_label_y), text, font=regular_font, fill="black")
        draw.line(
            (x + 5, sign_label_y + 50, x + 5 + max_line_width, sign_label_y + 50),
            fill="black",
            width=2,
        )

    # Observação
    obs_text = (
        "OBSERVAÇÃO: Nos termos da Convenção Coletiva a diária de viagem é destinada apenas ao colaborador "
        "que exercer atividade fora da base considerando cada período modular de 24 horas, o recebimento da diária "
        "exclui-se o pagamento da ajuda de alimentação (Ticket)."
    )
    obs_y = sign_label_y + 90
    obs_line_height = (
        getattr(small_font, "size", 24) + 8
        if hasattr(small_font, "size")
        else 32
    )
    draw_wrapped_text(
        draw,
        obs_text,
        margin,
        obs_y,
        small_font,
        width - 2 * margin,
        obs_line_height,
    )
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def sum_total(rows: list[dict]) -> float:
    """Sum currency columns for total display."""
    total = 0.0
    for row in rows:
        for key in ("DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO", "VALOR_HOTEL"):
            total += to_float(row.get(key, 0))
    return total


def build_table_dataframe(tipo: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Generate a dataframe structure with the required columns."""
    column_order = [
        "DATA",
        "CIDADE",
        *(["HOTEL", "VALOR_HOTEL"] if tipo == "AJUDANTE" else []),
        "DIARIA_EM_VIAGEM",
        "TICKET_ALIMENTACAO",
    ]
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
    df = pd.DataFrame(rows, columns=column_order)
    return df


def ensure_session_state() -> None:
    """Prepare session state keys used across the app."""
    st.session_state.setdefault("rdv_table", None)
    st.session_state.setdefault("rdv_meta", {})
    st.session_state.setdefault("generated_image_data", None)
    st.session_state.setdefault("generated_image_name", "")
    st.session_state.setdefault("generated_image_page", "")
    st.session_state.setdefault("batch_previews", {"tipo": "", "previews": []})


def open_print_window(image_data: bytes) -> None:
    """Open a new browser tab with the RDV image and call window.print()."""
    _open_print_window([image_data])


def open_print_window_batch(images: list[bytes]) -> None:
    """Open a new browser tab with multiple RDV images for batch printing."""
    if not images:
        return
    _open_print_window(images)


def _open_print_window(images: list[bytes]) -> None:
    encoded_images = [base64.b64encode(img).decode("ascii") for img in images]
    imgs_markup = "".join(
        f'<div class="page"><img src="data:image/png;base64,{encoded}" /></div>'
        for encoded in encoded_images
    )
    components.html(
        f"""
        <script>
            const printWindow = window.open("", "_blank");
            if (printWindow) {{
                printWindow.document.write(`
                    <!DOCTYPE html>
                    <html>
                        <head>
                            <meta charset="utf-8" />
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
                                    background: #ffffff;
                                }}
                                .page {{
                                    width: 100%;
                                    page-break-after: always;
                                }}
                                img {{
                                    width: 100%;
                                    height: auto;
                                    display: block;
                                }}
                            </style>
                        </head>
                        <body>
                            {imgs_markup}
                        </body>
                    </html>
                `);
                printWindow.document.close();
                printWindow.focus();
                printWindow.print();
            }} else {{
                alert("Permitida a abertura de janelas pop-up para imprimir o RDV.");
            }}
        </script>
        """,
        height=0,
    )


def render_generated_image(page: str) -> None:
    """Display the most recently generated image if it belongs to this page."""
    image_data = st.session_state.get("generated_image_data")
    image_page = st.session_state.get("generated_image_page")
    if image_data and page == image_page:
        st.image(image_data, use_container_width=True)
        st.download_button(
            "Baixar RDV (PNG)",
            data=image_data,
            file_name=st.session_state.get("generated_image_name", "rdv.png"),
            mime="image/png",
        )
        if st.button("Imprimir RDV gerado", key=f"print_{page}"):
            open_print_window(image_data)


def main() -> None:
    """Primary Streamlit app logic, with sidebar navigation."""
    ensure_session_state()
    init_db()
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(LOGO_PATH, width=80)
        else:
            st.markdown("JR")
    with title_col:
        st.title("Relatório de Despesas de Viagem – RDV JR")
    page = st.sidebar.selectbox(
        "Menu", ["Novo RDV", "Relatórios salvos"]
    )

    if page == "Novo RDV":
        st.header("Novo RDV")
        tipo = st.selectbox("Tipo de colaborador", TIPOS_COLABORADOR, format_func=lambda t: "Motorista" if t == "MOTORISTA" else "Ajudante")
        colaboradores_disponiveis = listar_colaboradores_por_tipo(tipo)
        if not colaboradores_disponiveis:
            st.warning("Não há colaboradores cadastrados para o tipo selecionado.")
            return
        modo_all_label = f"Todos os {tipo.capitalize()}s"
        modo = st.radio(
            "Modo de geração",
            ["Individual", modo_all_label],
            horizontal=True,
        )
        default_start, default_end = get_default_quinzena()
        data_inicial = st.date_input("Data inicial", value=default_start)
        data_final = st.date_input("Data final", value=default_end)
        if data_final < data_inicial:
            st.error("A data final deve ser igual ou posterior à data inicial.")
            return
        adiantamento_flag = st.checkbox("Houve adiantamento de diária?")
        valor_adiantamento = 0.0
        if adiantamento_flag:
            valor_adiantamento = st.number_input(
                "Valor do adiantamento (R$)",
                min_value=0.0,
                value=0.0,
                step=10.0,
                format="%f",
            )

        if st.session_state["batch_previews"].get("tipo") != tipo:
            st.session_state["batch_previews"] = {"tipo": "", "previews": []}

        if modo == "Individual":
            nomes_colabs = [c["nome"] for c in colaboradores_disponiveis]
            nome_escolhido = st.selectbox("Colaborador", nomes_colabs)
            colaborador = next(c for c in colaboradores_disponiveis if c["nome"] == nome_escolhido)
            if (
                st.session_state["rdv_meta"].get("tipo") != tipo
                or st.session_state["rdv_meta"].get("colaborador_nome") != colaborador["nome"]
            ):
                st.session_state["rdv_table"] = None
                st.session_state["rdv_meta"] = {}

            if st.button("Gerar tabela da quinzena"):
                df = build_table_dataframe(tipo, data_inicial, data_final)
                st.session_state["rdv_table"] = df
                st.session_state["rdv_meta"] = {
                    "tipo": tipo,
                    "colaborador_nome": colaborador["nome"],
                    "data_inicial": data_inicial,
                    "data_final": data_final,
                    "adiantamento": adiantamento_flag,
                    "valor_adiantamento": valor_adiantamento,
                }

            if st.session_state["rdv_table"] is not None:
                df = st.session_state["rdv_table"]
                column_config = {
                    "DATA": st.column_config.Column("Data", disabled=True),
                }
                st.session_state["rdv_table"] = st.data_editor(
                    df,
                    column_config=column_config,
                    use_container_width=True,
                )
                df = st.session_state["rdv_table"]
                total = sum_total(df.to_dict("records"))
                st.markdown(f"**TOTAL DA QUINZENA EM R$:** {format_currency(total)}")
                if st.button("Salvar RDV"):
                    if df.empty:
                        st.error("A tabela da quinzena está vazia.")
                    else:
                        linhas = []
                        for _, row in df.iterrows():
                            linhas.append(
                                {
                                    "DATA": datetime.strptime(row["DATA"], "%d/%m/%Y").date().isoformat(),
                                    "CIDADE": row.get("CIDADE", ""),
                                    "HOTEL": row.get("HOTEL") if tipo == "AJUDANTE" else None,
                                    "VALOR_HOTEL": to_float(row.get("VALOR_HOTEL", 0)) if tipo == "AJUDANTE" else None,
                                    "DIARIA_EM_VIAGEM": to_float(row.get("DIARIA_EM_VIAGEM", 0)),
                                    "TICKET_ALIMENTACAO": to_float(row.get("TICKET_ALIMENTACAO", 0)),
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
                        st.success("RDV salvo com sucesso.")
                        st.session_state["rdv_table"] = None
                        st.session_state["rdv_meta"] = {}
                if st.button("Gerar imagem do RDV (PNG)"):
                    meta = st.session_state["rdv_meta"]
                    if not meta:
                        st.error("Gere a tabela antes de criar a imagem.")
                    else:
                        df_rows = df.to_dict("records")
                        img_buffer = generate_image(
                            colaborador_nome=colaborador["nome"],
                            tipo=tipo,
                            data_inicial=data_inicial,
                            data_final=data_final,
                            adiantamento=adiantamento_flag,
                            valor_adiantamento=valor_adiantamento if adiantamento_flag else 0.0,
                            linhas=df_rows,
                            mostrar_valores=True,
                        )
                        st.session_state["generated_image_data"] = img_buffer.getvalue()
                        st.session_state["generated_image_name"] = (
                            f"RDV_{colaborador['nome'].replace(' ', '_')}_{tipo}_"
                            f"{data_inicial.strftime('%Y%m%d')}-{data_final.strftime('%Y%m%d')}.png"
                        )
                        st.session_state["generated_image_page"] = page
            render_generated_image(page)
        else:
            st.session_state["rdv_table"] = None
            st.session_state["rdv_meta"] = {}
            if st.button(f"Gerar RDVs para todos os {tipo.lower()}s"):
                df_template = build_table_dataframe(tipo, data_inicial, data_final)
                rows_template = df_template.to_dict("records")
                previews = []
                for colab in colaboradores_disponiveis:
                    rows_copy = [row.copy() for row in rows_template]
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
                            "file_name": (
                                f"RDV_{colab['nome'].replace(' ', '_')}_{tipo}_"
                                f"{data_inicial.strftime('%Y%m%d')}-{data_final.strftime('%Y%m%d')}.png"
                            ),
                        }
                    )
                st.session_state["batch_previews"] = {
                    "tipo": tipo,
                    "data_inicial": data_inicial,
                    "data_final": data_final,
                    "previews": previews,
                }
                st.success(f"Gerados {len(previews)} RDVs para {tipo.lower()}s.")
            batch_data = st.session_state.get("batch_previews", {})
            if batch_data.get("tipo") == tipo:
                for idx, preview in enumerate(batch_data["previews"]):
                    st.subheader(preview["nome"])
                    st.image(preview["image"], use_container_width=True)
                    st.download_button(
                        "Baixar RDV (PNG)",
                        data=preview["image"],
                        file_name=preview["file_name"],
                        mime="image/png",
                        key=f"batch_download_{tipo}_{idx}",
                    )
                    if st.button("Imprimir RDV", key=f"batch_print_{tipo}_{idx}"):
                        open_print_window(preview["image"])
                if batch_data["previews"]:
                    if st.button(f"Imprimir todos os {tipo.lower()}s", key=f"batch_print_all_{tipo}"):
                        open_print_window_batch(
                            [preview["image"] for preview in batch_data["previews"]]
                        )
    else:  # Relatórios salvos
        # Listagem e reabertura de RDVs previamente salvos
        st.header("Relatórios salvos")
        rdvs = get_rdvs()
        if not rdvs:
            st.info("Nenhum RDV salvo no banco.")
            return
        rdv_label = {
            f"{item['nome']} | {item['tipo']} | {format_date_br(item['data_inicial'])} a {format_date_br(item['data_final'])} (ID: {item['id']})": item
            for item in rdvs
        }
        selecionado = st.selectbox("RDV", options=list(rdv_label.keys()))
        rdv_data = rdv_label[selecionado]
        linhas = get_rdv_linhas(rdv_data["id"])
        st.markdown(
            f"**Colaborador:** {rdv_data['nome']} | **Tipo:** {rdv_data['tipo']} | "
            f"**Quinzena:** {format_date_br(rdv_data['data_inicial'])} a {format_date_br(rdv_data['data_final'])}"
        )
        st.markdown(
            f"**Total da quinzena:** {format_currency(rdv_data['total_quinzena'])} | "
            f"**Adiantamento:** {'Sim' if rdv_data['adiantamento'] else 'Não'} "
            f"{format_currency(rdv_data['valor_adiantamento']) if rdv_data['adiantamento'] else ''}"
        )
        df = pd.DataFrame(linhas)
        if rdv_data["tipo"] == "MOTORISTA":
            st.dataframe(df[["DATA", "CIDADE", "DIARIA_EM_VIAGEM", "TICKET_ALIMENTACAO"]])
        else:
            st.dataframe(
                df[
                    [
                        "DATA",
                        "CIDADE",
                        "HOTEL",
                        "VALOR_HOTEL",
                        "DIARIA_EM_VIAGEM",
                        "TICKET_ALIMENTACAO",
                    ]
                ]
            )
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
            st.session_state["generated_image_page"] = page
        render_generated_image(page)


if __name__ == "__main__":
    main()
