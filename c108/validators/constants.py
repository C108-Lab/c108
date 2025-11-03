"""
Data Validators Constants
"""


class CountryCodes:
    """
    Country and region code constants for validation.

    Provides official ISO 3166-1 country codes for validating country/region
    identifiers in addresses, locales, phone numbers, and geographic data.
    Used by validators to verify country codes against international standards.

    Intended for lookup/validation only.

    Attributes:
        ISO_3166_1_CODES: Set of valid ISO 3166-1 alpha-2 country codes.
            Two-letter codes for countries, dependencies, and special areas.
            Standard: ISO 3166-1:2020 [[2]](https://en.wikipedia.org/wiki/ISO_3166-1)
            Examples: {'US', 'GB', 'FR', 'DE', 'JP', 'CN', 'CA', ...}
            Includes territories and special regions (e.g., 'AQ' for Antarctica).
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


class LanguageCodes:
    """
    Language and script code constants for validation.

    Provides official code sets from ISO standards for validating language codes
    and writing system (script) identifiers. Used by language validation functions
    to verify codes against international standards.

    Intended for lookup/validation only.

    Attributes:
        ISO_639_1_CODES: Set of valid ISO 639-1 two-letter language codes.
            Covers 184 major world languages with standardized identifiers.
            Standard: ISO 639-1:2002 (https://datahub.io/core/language-codes)
            Examples: {'en', 'fr', 'de', 'ja', 'zh', 'ar', ...}

        ISO_15924_CODES: Set of valid ISO 15924 four-letter script codes.
            Identifies writing systems/scripts (210+ unique scripts).
            Standard: ISO 15924:2022 (https://localizely.com/iso-15924-list/)
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
