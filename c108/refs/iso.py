"""
ISO standard code sets for geographic, linguistic, and financial reference data.

Provides static frozensets for fast membership checks against official ISO standards.
Intended as reference data for validation logic, dataset pipelines, and ML workflows.

Classes:
    CountryCodes  – ISO 3166-1 alpha-2 country codes
    CurrencyCodes – ISO 4217 currency codes
    LanguageCodes – ISO 639-1 language codes and ISO 15924 script codes
"""


class CountryCodes:
    """
    ISO 3166-1 alpha-2 country and territory codes.

    Attributes:
        ISO_3166_1_CODES: Two-letter codes for countries, dependencies, and special areas.
            Standard: ISO 3166-1:2020 (https://en.wikipedia.org/wiki/ISO_3166-1)
            Examples: {'us', 'gb', 'fr', 'de', 'jp', 'cn', 'ca', ...}
            Includes territories and special regions (e.g., 'aq' for Antarctica).
    """

    ISO_3166_1_CODES = {
        "ad",  # Andorra
        "ae",  # United Arab Emirates
        "af",  # Afghanistan
        "ag",  # Antigua and Barbuda
        "ai",  # Anguilla
        "al",  # Albania
        "am",  # Armenia
        "ao",  # Angola
        "aq",  # Antarctica
        "ar",  # Argentina
        "as",  # American Samoa
        "at",  # Austria
        "au",  # Australia
        "aw",  # Aruba
        "ax",  # Åland Islands
        "az",  # Azerbaijan
        "ba",  # Bosnia and Herzegovina
        "bb",  # Barbados
        "bd",  # Bangladesh
        "be",  # Belgium
        "bf",  # Burkina Faso
        "bg",  # Bulgaria
        "bh",  # Bahrain
        "bi",  # Burundi
        "bj",  # Benin
        "bl",  # Saint Barthélemy
        "bm",  # Bermuda
        "bn",  # Brunei Darussalam
        "bo",  # Bolivia
        "bq",  # Bonaire, Sint Eustatius and Saba
        "br",  # Brazil
        "bs",  # Bahamas
        "bt",  # Bhutan
        "bv",  # Bouvet Island
        "bw",  # Botswana
        "by",  # Belarus
        "bz",  # Belize
        "ca",  # Canada
        "cc",  # Cocos (Keeling) Islands
        "cd",  # Congo, Democratic Republic of the
        "cf",  # Central African Republic
        "cg",  # Congo
        "ch",  # Switzerland
        "ci",  # Côte d'Ivoire
        "ck",  # Cook Islands
        "cl",  # Chile
        "cm",  # Cameroon
        "cn",  # China
        "co",  # Colombia
        "cr",  # Costa Rica
        "cu",  # Cuba
        "cv",  # Cabo Verde
        "cw",  # Curaçao
        "cx",  # Christmas Island
        "cy",  # Cyprus
        "cz",  # Czechia
        "de",  # Germany
        "dj",  # Djibouti
        "dk",  # Denmark
        "dm",  # Dominica
        "do",  # Dominican Republic
        "dz",  # Algeria
        "ec",  # Ecuador
        "ee",  # Estonia
        "eg",  # Egypt
        "eh",  # Western Sahara
        "er",  # Eritrea
        "es",  # Spain
        "et",  # Ethiopia
        "fi",  # Finland
        "fj",  # Fiji
        "fk",  # Falkland Islands (Malvinas)
        "fm",  # Micronesia, Federated States of
        "fo",  # Faroe Islands
        "fr",  # France
        "ga",  # Gabon
        "gb",  # United Kingdom
        "gd",  # Grenada
        "ge",  # Georgia
        "gf",  # French Guiana
        "gg",  # Guernsey
        "gh",  # Ghana
        "gi",  # Gibraltar
        "gl",  # Greenland
        "gm",  # Gambia
        "gn",  # Guinea
        "gp",  # Guadeloupe
        "gq",  # Equatorial Guinea
        "gr",  # Greece
        "gs",  # South Georgia and the South Sandwich Islands
        "gt",  # Guatemala
        "gu",  # Guam
        "gw",  # Guinea-Bissau
        "gy",  # Guyana
        "hk",  # Hong Kong
        "hm",  # Heard Island and McDonald Islands
        "hn",  # Honduras
        "hr",  # Croatia
        "ht",  # Haiti
        "hu",  # Hungary
        "id",  # Indonesia
        "ie",  # Ireland
        "il",  # Israel
        "im",  # Isle of Man
        "in",  # India
        "io",  # British Indian Ocean Territory
        "iq",  # Iraq
        "ir",  # Iran, Islamic Republic of
        "is",  # Iceland
        "it",  # Italy
        "je",  # Jersey
        "jm",  # Jamaica
        "jo",  # Jordan
        "jp",  # Japan
        "ke",  # Kenya
        "kg",  # Kyrgyzstan
        "kh",  # Cambodia
        "ki",  # Kiribati
        "km",  # Comoros
        "kn",  # Saint Kitts and Nevis
        "kp",  # Korea, Democratic People's Republic of
        "kr",  # Korea, Republic of
        "kw",  # Kuwait
        "ky",  # Cayman Islands
        "kz",  # Kazakhstan
        "la",  # Lao People's Democratic Republic
        "lb",  # Lebanon
        "lc",  # Saint Lucia
        "li",  # Liechtenstein
        "lk",  # Sri Lanka
        "lr",  # Liberia
        "ls",  # Lesotho
        "lt",  # Lithuania
        "lu",  # Luxembourg
        "lv",  # Latvia
        "ly",  # Libya
        "ma",  # Morocco
        "mc",  # Monaco
        "md",  # Moldova, Republic of
        "me",  # Montenegro
        "mf",  # Saint Martin (French part)
        "mg",  # Madagascar
        "mh",  # Marshall Islands
        "mk",  # North Macedonia
        "ml",  # Mali
        "mm",  # Myanmar
        "mn",  # Mongolia
        "mo",  # Macao
        "mp",  # Northern Mariana Islands
        "mq",  # Martinique
        "mr",  # Mauritania
        "ms",  # Montserrat
        "mt",  # Malta
        "mu",  # Mauritius
        "mv",  # Maldives
        "mw",  # Malawi
        "mx",  # Mexico
        "my",  # Malaysia
        "mz",  # Mozambique
        "na",  # Namibia
        "nc",  # New Caledonia
        "ne",  # Niger
        "nf",  # Norfolk Island
        "ng",  # Nigeria
        "ni",  # Nicaragua
        "nl",  # Netherlands
        "no",  # Norway
        "np",  # Nepal
        "nr",  # Nauru
        "nu",  # Niue
        "nz",  # New Zealand
        "om",  # Oman
        "pa",  # Panama
        "pe",  # Peru
        "pf",  # French Polynesia
        "pg",  # Papua New Guinea
        "ph",  # Philippines
        "pk",  # Pakistan
        "pl",  # Poland
        "pm",  # Saint Pierre and Miquelon
        "pn",  # Pitcairn
        "pr",  # Puerto Rico
        "ps",  # Palestine, State of
        "pt",  # Portugal
        "pw",  # Palau
        "py",  # Paraguay
        "qa",  # Qatar
        "re",  # Réunion
        "ro",  # Romania
        "rs",  # Serbia
        "ru",  # Russian Federation
        "rw",  # Rwanda
        "sa",  # Saudi Arabia
        "sb",  # Solomon Islands
        "sc",  # Seychelles
        "sd",  # Sudan
        "se",  # Sweden
        "sg",  # Singapore
        "sh",  # Saint Helena, Ascension and Tristan da Cunha
        "si",  # Slovenia
        "sj",  # Svalbard and Jan Mayen
        "sk",  # Slovakia
        "sl",  # Sierra Leone
        "sm",  # San Marino
        "sn",  # Senegal
        "so",  # Somalia
        "sr",  # Suriname
        "ss",  # South Sudan
        "st",  # Sao Tome and Principe
        "sv",  # El Salvador
        "sx",  # Sint Maarten (Dutch part)
        "sy",  # Syrian Arab Republic
        "sz",  # Eswatini
        "tc",  # Turks and Caicos Islands
        "td",  # Chad
        "tf",  # French Southern Territories
        "tg",  # Togo
        "th",  # Thailand
        "tj",  # Tajikistan
        "tk",  # Tokelau
        "tl",  # Timor-Leste
        "tm",  # Turkmenistan
        "tn",  # Tunisia
        "to",  # Tonga
        "tr",  # Turkey
        "tt",  # Trinidad and Tobago
        "tv",  # Tuvalu
        "tw",  # Taiwan, Province of China
        "tz",  # Tanzania, United Republic of
        "ua",  # Ukraine
        "ug",  # Uganda
        "um",  # United States Minor Outlying Islands
        "us",  # United States of America
        "uy",  # Uruguay
        "uz",  # Uzbekistan
        "va",  # Holy See (Vatican City State)
        "vc",  # Saint Vincent and the Grenadines
        "ve",  # Venezuela, Bolivarian Republic of
        "vg",  # Virgin Islands, British
        "vi",  # Virgin Islands, U.S.
        "vn",  # Viet Nam
        "vu",  # Vanuatu
        "wf",  # Wallis and Futuna
        "ws",  # Samoa
        "ye",  # Yemen
        "yt",  # Mayotte
        "za",  # South Africa
        "zm",  # Zambia
        "zw",  # Zimbabwe
    }


# **NOTE on Currencies as up-to-date ISO 4217 in year 2026**
#
# * `"sle"` — Sierra Leonean leone. The ISO 4217 standard replaced `SLL` with `SLE` in 2022 after a redenomination.
#   Just worth being aware of if you need to support legacy data using `SLL`.
#
# **A few things worth noting (not errors, but FYI):**
#
# * `"hrk"` (Croatian kuna) is absent — Croatia adopted the euro (`EUR`) in January 2023, so its omission is correct.
# * `"cuc"` (Cuban convertible peso) is absent — Cuba officially unified its currency in 2021,
#   so omitting it is reasonable for current standards, though some data sources still reference it.
# * `"stb"` / `"std"` (old São Tomé dobra) is absent — replaced by `"stn"` in 2018.
# * `"zwl"` (Zimbabwe dollar) is absent — replaced by `"zwg"` (Zimbabwe Gold) in 2024.


class CurrencyCodes:
    """
    ISO 4217 three-letter currency codes.

    Attributes:
        ISO_4217_CODES: Active currencies, supranational currencies, precious metals,
            and special allocation codes.
            Standard: ISO 4217 (https://en.wikipedia.org/wiki/ISO_4217)
            Examples: {'usd', 'eur', 'jpy', 'gbp', 'chf', 'aud', ...}
    """

    ISO_4217_CODES = {
        "aed",  # United Arab Emirates dirham
        "afn",  # Afghan afghani
        "all",  # Albanian lek
        "amd",  # Armenian dram
        "aoa",  # Angolan kwanza
        "ars",  # Argentine peso
        "aud",  # Australian dollar
        "awg",  # Aruban florin
        "azn",  # Azerbaijani manat
        "bam",  # Bosnia and Herzegovina convertible mark
        "bbd",  # Barbados dollar
        "bdt",  # Bangladeshi taka
        "bhd",  # Bahraini dinar
        "bif",  # Burundian franc
        "bmd",  # Bermudian dollar
        "bnd",  # Brunei dollar
        "bob",  # Boliviano
        "bov",  # Bolivian Mvdol (funds code)
        "brl",  # Brazilian real
        "bsd",  # Bahamian dollar
        "btn",  # Bhutanese ngultrum
        "bwp",  # Botswana pula
        "byn",  # Belarusian ruble
        "bzd",  # Belize dollar
        "cad",  # Canadian dollar
        "cdf",  # Congolese franc
        "che",  # WIR euro (complementary currency)
        "chf",  # Swiss franc
        "chw",  # WIR franc (complementary currency)
        "clf",  # Unidad de Fomento (funds code)
        "clp",  # Chilean peso
        "cny",  # Renminbi (Chinese yuan)
        "cop",  # Colombian peso
        "cou",  # Unidad de Valor Real (UVR) (funds code)
        "crc",  # Costa Rican colon
        "cup",  # Cuban peso
        "cve",  # Cape Verdean escudo
        "czk",  # Czech koruna
        "djf",  # Djiboutian franc
        "dkk",  # Danish krone
        "dop",  # Dominican peso
        "dzd",  # Algerian dinar
        "egp",  # Egyptian pound
        "ern",  # Eritrean nakfa
        "etb",  # Ethiopian birr
        "eur",  # Euro
        "fjd",  # Fiji dollar
        "fkp",  # Falkland Islands pound
        "gbp",  # Pound sterling
        "gel",  # Georgian lari
        "ghs",  # Ghanaian cedi
        "gip",  # Gibraltar pound
        "gmd",  # Gambian dalasi
        "gnf",  # Guinean franc
        "gtq",  # Guatemalan quetzal
        "gyd",  # Guyanese dollar
        "hkd",  # Hong Kong dollar
        "hnl",  # Honduran lempira
        "htg",  # Haitian gourde
        "huf",  # Hungarian forint
        "idr",  # Indonesian rupiah
        "ils",  # Israeli new shekel
        "inr",  # Indian rupee
        "iqd",  # Iraqi dinar
        "irr",  # Iranian rial
        "isk",  # Icelandic króna
        "jmd",  # Jamaican dollar
        "jod",  # Jordanian dinar
        "jpy",  # Japanese yen
        "kes",  # Kenyan shilling
        "kgs",  # Kyrgyzstani som
        "khr",  # Cambodian riel
        "kmf",  # Comorian franc
        "kpw",  # North Korean won
        "krw",  # South Korean won
        "kwd",  # Kuwaiti dinar
        "kyd",  # Cayman Islands dollar
        "kzt",  # Kazakhstani tenge
        "lak",  # Lao kip
        "lbp",  # Lebanese pound
        "lkr",  # Sri Lankan rupee
        "lrd",  # Liberian dollar
        "lsl",  # Lesotho loti
        "lyd",  # Libyan dinar
        "mad",  # Moroccan dirham
        "mdl",  # Moldovan leu
        "mga",  # Malagasy ariary
        "mkd",  # Macedonian denar
        "mmk",  # Myanmar kyat
        "mnt",  # Mongolian tögrög
        "mop",  # Macanese pataca
        "mru",  # Mauritanian ouguiya
        "mur",  # Mauritian rupee
        "mvr",  # Maldivian rufiyaa
        "mwk",  # Malawian kwacha
        "mxn",  # Mexican peso
        "mxv",  # Mexican Unidad de Inversion (UDI) (funds code)
        "myr",  # Malaysian ringgit
        "mzn",  # Mozambican metical
        "nad",  # Namibian dollar
        "ngn",  # Nigerian naira
        "nio",  # Nicaraguan córdoba
        "nok",  # Norwegian krone
        "npr",  # Nepalese rupee
        "nzd",  # New Zealand dollar
        "omr",  # Omani rial
        "pab",  # Panamanian balboa
        "pen",  # Peruvian sol
        "pgk",  # Papua New Guinean kina
        "php",  # Philippine peso
        "pkr",  # Pakistani rupee
        "pln",  # Polish złoty
        "pyg",  # Paraguayan guaraní
        "qar",  # Qatari riyal
        "ron",  # Romanian leu
        "rsd",  # Serbian dinar
        "rub",  # Russian ruble
        "rwf",  # Rwandan franc
        "sar",  # Saudi riyal
        "sbd",  # Solomon Islands dollar
        "scr",  # Seychelles rupee
        "sdg",  # Sudanese pound
        "sek",  # Swedish krona
        "sgd",  # Singapore dollar
        "shp",  # Saint Helena pound
        "sle",  # Sierra Leonean leone
        "sos",  # Somali shilling
        "srd",  # Surinamese dollar
        "ssp",  # South Sudanese pound
        "stn",  # São Tomé and Príncipe dobra
        "svc",  # Salvadoran colón
        "syp",  # Syrian pound
        "szl",  # Swazi lilangeni
        "thb",  # Thai baht
        "tjs",  # Tajikistani somoni
        "tmt",  # Turkmenistan manat
        "tnd",  # Tunisian dinar
        "top",  # Tongan paʻanga
        "try",  # Turkish lira
        "ttd",  # Trinidad and Tobago dollar
        "twd",  # New Taiwan dollar
        "tzs",  # Tanzanian shilling
        "uah",  # Ukrainian hryvnia
        "ugx",  # Ugandan shilling
        "usd",  # United States dollar
        "usn",  # United States dollar (next day) (funds code)
        "uyi",  # Uruguay Peso en Unidades Indexadas (URUIURUI) (funds code)
        "uyu",  # Uruguayan peso
        "uyw",  # Unidad Previsional
        "uzs",  # Uzbekistan som
        "ved",  # Venezuelan bolívar digital
        "ves",  # Venezuelan bolívar soberano
        "vnd",  # Vietnamese đồng
        "vuv",  # Vanuatu vatu
        "wst",  # Samoan tala
        "xaf",  # CFA franc BEAC
        "xag",  # Silver (one troy ounce)
        "xau",  # Gold (one troy ounce)
        "xba",  # European Composite Unit (EURCO) (bond market unit)
        "xbb",  # European Monetary Unit (E.M.U.-6) (bond market unit)
        "xbc",  # European Unit of Account 9 (E.U.A.-9) (bond market unit)
        "xbd",  # European Unit of Account 17 (E.U.A.-17) (bond market unit)
        "xcd",  # East Caribbean dollar
        "xcg",  # Caribbean guilder
        "xdr",  # Special drawing rights
        "xof",  # CFA franc BCEAO
        "xpd",  # Palladium (one troy ounce)
        "xpf",  # CFP franc (franc Pacifique)
        "xpt",  # Platinum (one troy ounce)
        "xsu",  # SUCRE
        "xts",  # Code reserved for testing
        "xua",  # ADB Unit of Account
        "xxx",  # No currency
        "yer",  # Yemeni rial
        "zar",  # South African rand
        "zmw",  # Zambian kwacha
        "zwg",  # Zimbabwe Gold
    }


class LanguageCodes:
    """
    ISO language and script code sets.

    Attributes:
        ISO_639_1_CODES: Two-letter language codes for 184 major world languages.
            Standard: ISO 639-1:2002 (https://datahub.io/core/language-codes)
            Examples: {'en', 'fr', 'de', 'ja', 'zh', 'ar', ...}

        ISO_15924_CODES: Four-letter script codes identifying 210+ writing systems.
            Standard: ISO 15924:2022 (https://unicode.org/iso15924/)
            Examples: {'Latn', 'Cyrl', 'Arab', 'Hans', 'Hant', 'Deva', ...}
    """

    ISO_639_1_CODES = {
        "aa",  # Afar
        "ab",  # Abkhazian
        "ae",  # Avestan
        "af",  # Afrikaans
        "ak",  # Akan
        "am",  # Amharic
        "an",  # Aragonese
        "ar",  # Arabic
        "as",  # Assamese
        "av",  # Avaric
        "ay",  # Aymara
        "az",  # Azerbaijani
        "ba",  # Bashkir
        "be",  # Belarusian
        "bg",  # Bulgarian
        "bh",  # Bihari languages
        "bi",  # Bislama
        "bm",  # Bambara
        "bn",  # Bengali
        "bo",  # Tibetan
        "br",  # Breton
        "bs",  # Bosnian
        "ca",  # Catalan
        "ce",  # Chechen
        "ch",  # Chamorro
        "co",  # Corsican
        "cr",  # Cree
        "cs",  # Czech
        "cu",  # Church Slavic
        "cv",  # Chuvash
        "cy",  # Welsh
        "da",  # Danish
        "de",  # German
        "dv",  # Divehi
        "dz",  # Dzongkha
        "ee",  # Ewe
        "el",  # Greek
        "en",  # English
        "eo",  # Esperanto
        "es",  # Spanish
        "et",  # Estonian
        "eu",  # Basque
        "fa",  # Persian
        "ff",  # Fulah
        "fi",  # Finnish
        "fj",  # Fijian
        "fo",  # Faroese
        "fr",  # French
        "fy",  # Western Frisian
        "ga",  # Irish
        "gd",  # Gaelic
        "gl",  # Galician
        "gn",  # Guarani
        "gu",  # Gujarati
        "gv",  # Manx
        "ha",  # Hausa
        "he",  # Hebrew
        "hi",  # Hindi
        "ho",  # Hiri Motu
        "hr",  # Croatian
        "ht",  # Haitian
        "hu",  # Hungarian
        "hy",  # Armenian
        "hz",  # Herero
        "ia",  # Interlingua
        "id",  # Indonesian
        "ie",  # Interlingue
        "ig",  # Igbo
        "ii",  # Sichuan Yi
        "ik",  # Inupiaq
        "io",  # Ido
        "is",  # Icelandic
        "it",  # Italian
        "iu",  # Inuktitut
        "ja",  # Japanese
        "jv",  # Javanese
        "ka",  # Georgian
        "kg",  # Kongo
        "ki",  # Kikuyu
        "kj",  # Kuanyama
        "kk",  # Kazakh
        "kl",  # Kalaallisut
        "km",  # Central Khmer
        "kn",  # Kannada
        "ko",  # Korean
        "kr",  # Kanuri
        "ks",  # Kashmiri
        "ku",  # Kurdish
        "kv",  # Komi
        "kw",  # Cornish
        "ky",  # Kirghiz
        "la",  # Latin
        "lb",  # Luxembourgish
        "lg",  # Ganda
        "li",  # Limburgan
        "ln",  # Lingala
        "lo",  # Lao
        "lt",  # Lithuanian
        "lu",  # Luba-Katanga
        "lv",  # Latvian
        "mg",  # Malagasy
        "mh",  # Marshallese
        "mi",  # Maori
        "mk",  # Macedonian
        "ml",  # Malayalam
        "mn",  # Mongolian
        "mr",  # Marathi
        "ms",  # Malay
        "mt",  # Maltese
        "my",  # Burmese
        "na",  # Nauru
        "nb",  # Norwegian Bokmål
        "nd",  # North Ndebele
        "ne",  # Nepali
        "ng",  # Ndonga
        "nl",  # Dutch
        "nn",  # Norwegian Nynorsk
        "no",  # Norwegian
        "nr",  # South Ndebele
        "nv",  # Navajo
        "ny",  # Chichewa
        "oc",  # Occitan
        "oj",  # Ojibwa
        "om",  # Oromo
        "or",  # Oriya
        "os",  # Ossetian
        "pa",  # Panjabi
        "pi",  # Pali
        "pl",  # Polish
        "ps",  # Pushto
        "pt",  # Portuguese
        "qu",  # Quechua
        "rm",  # Romansh
        "rn",  # Rundi
        "ro",  # Romanian
        "ru",  # Russian
        "rw",  # Kinyarwanda
        "sa",  # Sanskrit
        "sc",  # Sardinian
        "sd",  # Sindhi
        "se",  # Northern Sami
        "sg",  # Sango
        "si",  # Sinhala
        "sk",  # Slovak
        "sl",  # Slovenian
        "sm",  # Samoan
        "sn",  # Shona
        "so",  # Somali
        "sq",  # Albanian
        "sr",  # Serbian
        "ss",  # Swati
        "st",  # Southern Sotho
        "su",  # Sundanese
        "sv",  # Swedish
        "sw",  # Swahili
        "ta",  # Tamil
        "te",  # Telugu
        "tg",  # Tajik
        "th",  # Thai
        "ti",  # Tigrinya
        "tk",  # Turkmen
        "tl",  # Tagalog
        "tn",  # Tswana
        "to",  # Tonga
        "tr",  # Turkish
        "ts",  # Tsonga
        "tt",  # Tatar
        "tw",  # Twi
        "ty",  # Tahitian
        "ug",  # Uighur
        "uk",  # Ukrainian
        "ur",  # Urdu
        "uz",  # Uzbek
        "ve",  # Venda
        "vi",  # Vietnamese
        "vo",  # Volapük
        "wa",  # Walloon
        "wo",  # Wolof
        "xh",  # Xhosa
        "yi",  # Yiddish
        "yo",  # Yoruba
        "za",  # Zhuang
        "zh",  # Chinese
        "zu",  # Zulu
    }
    ISO_15924_CODES = {
        "adlm",  # Adlam
        "afak",  # Afaka
        "aghb",  # Caucasian Albanian
        "ahom",  # Ahom, Tai Ahom
        "arab",  # Arabic
        "aran",  # Arabic (Nastaliq variant)
        "armi",  # Imperial Aramaic
        "armn",  # Armenian
        "avst",  # Avestan
        "bali",  # Balinese
        "bamu",  # Bamum
        "bass",  # Bassa Vah
        "batk",  # Batak
        "beng",  # Bengali (Bangla)
        "bhks",  # Bhaiksuki
        "blis",  # Blissymbols
        "bopo",  # Bopomofo
        "brah",  # Brahmi
        "brai",  # Braille
        "bugi",  # Buginese
        "buhd",  # Buhid
        "cakm",  # Chakma
        "cans",  # Unified Canadian Aboriginal Syllabics
        "cari",  # Carian
        "cham",  # Cham
        "cher",  # Cherokee
        "chrs",  # Chorasmian
        "cirt",  # Cirth
        "copt",  # Coptic
        "cpmn",  # Cypro-Minoan
        "cprt",  # Cypriot syllabary
        "cyrl",  # Cyrillic
        "cyrs",  # Cyrillic (Old Church Slavonic variant)
        "deva",  # Devanagari (Nagari)
        "diak",  # Dives Akuru
        "dogr",  # Dogra
        "dsrt",  # Deseret (Mormon)
        "dupl",  # Duployan shorthand, Duployan stenography
        "egyd",  # Egyptian demotic
        "egyh",  # Egyptian hieratic
        "egyp",  # Egyptian hieroglyphs
        "elba",  # Elbasan
        "elym",  # Elymaic
        "ethi",  # Ethiopic (Geʻez)
        "geok",  # Khutsuri (Asomtavruli and Nuskhuri)
        "geor",  # Georgian (Mkhedruli and Mtavruli)
        "glag",  # Glagolitic
        "gong",  # Gunjala Gondi
        "gonm",  # Masaram Gondi
        "goth",  # Gothic
        "gran",  # Grantha
        "grek",  # Greek
        "gujr",  # Gujarati
        "guru",  # Gurmukhi
        "hanb",  # Han with Bopomofo (alias for Han + Bopomofo)
        "hang",  # Hangul (Hangŭl, Hangeul)
        "hani",  # Han (Hanzi, Kanji, Hanja)
        "hano",  # Hanunoo (Hanunóo)
        "hans",  # Han (Simplified variant)
        "hant",  # Han (Traditional variant)
        "hatr",  # Hatran
        "hebr",  # Hebrew
        "hira",  # Hiragana
        "hluw",  # Anatolian Hieroglyphs (Luwian Hieroglyphs, Hittite Hieroglyphs)
        "hmng",  # Pahawh Hmong
        "hmnp",  # Nyiakeng Puachue Hmong
        "hrkt",  # Japanese syllabaries (alias for Hiragana + Katakana)
        "hung",  # Old Hungarian (Hungarian Runic)
        "inds",  # Indus (Harappan)
        "ital",  # Old Italic (Etruscan, Oscan, etc.)
        "jamo",  # Jamo (alias for Jamo subset of Hangul)
        "java",  # Javanese
        "jpan",  # Japanese (alias for Han + Hiragana + Katakana)
        "jurc",  # Jurchen
        "kali",  # Kayah Li
        "kana",  # Katakana
        "khar",  # Kharoshthi
        "khmr",  # Khmer
        "khoj",  # Khojki
        "kitl",  # Khitan large script
        "kits",  # Khitan small script
        "knda",  # Kannada
        "kore",  # Korean (alias for Hangul + Han)
        "kpel",  # Kpelle
        "kthi",  # Kaithi
        "lana",  # Tai Tham (Lanna)
        "laoo",  # Lao
        "latf",  # Latin (Fraktur variant)
        "latg",  # Latin (Gaelic variant)
        "latn",  # Latin
        "leke",  # Leke
        "lepc",  # Lepcha (Róng)
        "limb",  # Limbu
        "lina",  # Linear A
        "linb",  # Linear B
        "lisu",  # Lisu (Fraser)
        "loma",  # Loma
        "lyci",  # Lycian
        "lydi",  # Lydian
        "mahj",  # Mahajani
        "maka",  # Makasar
        "mand",  # Mandaic, Mandaean
        "mani",  # Manichaean
        "marc",  # Marchen
        "maya",  # Mayan hieroglyphs
        "medf",  # Medefaidrin (Oberi Okaime, Oberi Ɔkaimɛ)
        "mend",  # Mende Kikakui
        "merc",  # Meroitic Cursive
        "mero",  # Meroitic Hieroglyphs
        "mlym",  # Malayalam
        "modi",  # Modi, Moḍī
        "mong",  # Mongolian
        "moon",  # Moon (Moon code, Moon script, Moon type)
        "mroo",  # Mro, Mru
        "mtei",  # Meitei Mayek (Meithei, Meetei)
        "mult",  # Multani
        "mymr",  # Myanmar (Burmese)
        "nand",  # Nandinagari
        "narb",  # Old North Arabian (Ancient North Arabian)
        "nbat",  # Nabataean
        "newa",  # Newa, Newar, Newari, Nepāla lipi
        "nkdb",  # Naxi Dongba (na²¹ɕi³³ to³³ba²¹, Nakhi Tomba)
        "nkgb",  # Naxi Geba (na²¹ɕi³³ gʌ²¹ba²¹, 'Na-'Khi ²Ggŏ-¹baw, Nakhi Geba)
        "nkoo",  # N'Ko
        "nshu",  # Nüshu
        "ogam",  # Ogham
        "olck",  # Ol Chiki (Ol Cemet', Ol, Santali)
        "orkh",  # Old Turkic, Orkhon Runic
        "orya",  # Oriya (Odia)
        "osge",  # Osage
        "osma",  # Osmanya
        "palm",  # Palmyrene
        "pauc",  # Pau Cin Hau
        "perm",  # Old Permic
        "phag",  # Phags-pa
        "phli",  # Inscriptional Pahlavi
        "phlp",  # Psalter Pahlavi
        "phlv",  # Book Pahlavi
        "phnx",  # Phoenician
        "piqd",  # Klingon (KLI pIqaD)
        "plrd",  # Miao (Pollard)
        "prti",  # Inscriptional Parthian
        "qaaa",  # Reserved for private use (start)
        "qabx",  # Reserved for private use (end)
        "rjng",  # Rejang (Redjang, Kaganga)
        "rohg",  # Hanifi Rohingya
        "roro",  # Rongorongo
        "runr",  # Runic
        "samr",  # Samaritan
        "sara",  # Sarati
        "sarb",  # Old South Arabian
        "saur",  # Saurashtra
        "sgnw",  # SignWriting
        "shaw",  # Shavian (Shaw)
        "shrd",  # Sharada, Śāradā
        "shui",  # Shuishu
        "sidd",  # Siddham, Siddhaṃ, Siddhamātṛkā
        "sind",  # Khudawadi, Sindhi
        "sinh",  # Sinhala
        "sogd",  # Sogdian
        "sogo",  # Old Sogdian
        "sora",  # Sora Sompeng
        "soyo",  # Soyombo
        "sund",  # Sundanese
        "sylo",  # Syloti Nagri
        "syrc",  # Syriac
        "syre",  # Syriac (Estrangelo variant)
        "syrj",  # Syriac (Western variant)
        "syrn",  # Syriac (Eastern variant)
        "tagb",  # Tagbanwa
        "takr",  # Takri, Ṭākrī, Ṭāṅkrī
        "tale",  # Tai Le
        "talu",  # New Tai Lue
        "taml",  # Tamil
        "tang",  # Tangut
        "tavt",  # Tai Viet
        "telu",  # Telugu
        "teng",  # Tengwar
        "tfng",  # Tifinagh (Berber)
        "tglg",  # Tagalog (Baybayin, Alibata)
        "thaa",  # Thaana
        "thai",  # Thai
        "tibt",  # Tibetan
        "tirh",  # Tirhuta
        "tnsa",  # Tangsa
        "toto",  # Toto
        "ugar",  # Ugaritic
        "vaii",  # Vai
        "visp",  # Visible Speech
        "vith",  # Vithkuqi
        "wara",  # Warang Citi (Varang Kshiti)
        "wcho",  # Wancho
        "wole",  # Woleai
        "xpeo",  # Old Persian
        "xsux",  # Cuneiform, Sumero-Akkadian
        "yezi",  # Yezidi
        "yiii",  # Yi
        "zanb",  # Zanabazar Square (Zanabazarin Dörböljin Useg, Xewtee Dörböljin Bicig, Horizontal Square Script)
        "zinh",  # Code for inherited script
        "zmth",  # Mathematical notation
        "zsye",  # Symbols (Emoji variant)
        "zsym",  # Symbols
        "zxxx",  # Code for unwritten documents
        "zyyy",  # Code for undetermined script
        "zzzz",  # Code for uncoded script
    }
