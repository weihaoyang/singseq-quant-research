"""Exchange classifications for securities present in the supplied daily archive."""

# Each code appears as a B share in the SZSE market statistics fact book.
# https://docs.static.szse.cn/www/market/periodical/year/W020250912366937867370.pdf
SZSE_B_SHARE_CODES = (
    "200011", "200012", "200016", "200017", "200019", "200020", "200025", "200026",
    "200028", "200029", "200030", "200037", "200045", "200055", "200056", "200058",
    "200429", "200468", "200488", "200505", "200512", "200521", "200530", "200539",
    "200541", "200550", "200553", "200570", "200581", "200596", "200625", "200706",
    "200725", "200726", "200761", "200869", "200992", "201872",
)
SZSE_B_SHARE_SQL = ", ".join(f"'{code}'" for code in SZSE_B_SHARE_CODES)
SZSE_B_SHARE_SYMBOL_SQL = ", ".join(f"'{code}.SZ'" for code in SZSE_B_SHARE_CODES)
