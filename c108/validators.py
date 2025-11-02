"""
C108 Data Validators

This module contains validation functions for data formats and values
including emails, IP addresses, language codes, and URLs.

These validators check the **content/format** of values, not their types.
For runtime type validation, see the abc module.
"""

# Standard library -----------------------------------------------------------------------------------------------------
import ipaddress
import re

from typing import Literal

# Local ----------------------------------------------------------------------------------------------------------------
from .tools import fmt_type, fmt_value


# Constants ------------------------------------------------------------------------------------------------------------


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


# Methods --------------------------------------------------------------------------------------------------------------


def validate_email(email: str, *, strip: bool = True, lowercase: bool = True) -> str:
    """
    Validate email address format according to simplified RFC 5322 rules.

    This function performs basic email sanity checks for:
        - Non-empty input
        - Valid format (local-part@domain)
        - Length constraints (RFC 5321)
        - Character set compliance

    Note:
        This validation is intentionally simplified for lightweight use cases like
        form prototypes, data preprocessing, unit tests, and educational examples.
        It does NOT cover all RFC 5322 edge cases (quoted strings, comments,
        IP-literals, internationalized domains) or verify deliverability.

        For production systems requiring full RFC compliance, deliverability
        verification, or high-performance validation, use dedicated libraries:
            - `email-validator`: Full RFC compliance with DNS MX lookup support
            - `pydantic[email]`: Rust-backed validation for speed
            - SMTP verification libraries for deliverability checks (slow)

        While RFC 5321 technically allows case-sensitive local parts, virtually
        all modern email providers treat addresses case-insensitively. The default
        behavior normalizes to lowercase per community standards.

    Args:
        email: Email address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        lowercase: If True, convert email to lowercase after validation. Defaults
            to True. Recommended for consistency since email addresses are
            case-insensitive in practice.

    Returns:
        The validated email address. Transformations applied in order:
        1. Strip whitespace (if strip=True)
        2. Validate format
        3. Convert to lowercase (if lowercase=True)

    Raises:
        TypeError: If email is not a string.
        ValueError: If email is empty (after stripping if enabled), exceeds length
            limits (254 chars total, 64 for local part), or has invalid format.

    Examples:
        >>> validate_email("User@example.COM")
        'user@example.com'
        >>> validate_email("")
        Traceback (most recent call last):
            ...
        ValueError: email cannot be empty
        >>> validate_email("invalid.email")
        Traceback (most recent call last):
            ...
        ValueError: invalid email format: missing '@' symbol: 'invalid.email'
        >>> validate_email("a" * 65 + "@example.com")
        Traceback (most recent call last):
            ...
        ValueError: email local part exceeds maximum length of 64 characters (got 65)
    """

    # Type check
    if not isinstance(email, str):
        raise TypeError(f"Email must be a string, got {fmt_type(email)}")

    # Handle whitespace
    if strip:
        email = email.strip()
    elif email != email.strip():
        raise ValueError(
            f"invalid email format: contains leading or trailing whitespace: '{email}'"
        )

    # Check for empty
    if not email:
        raise ValueError("email cannot be empty")

    # Check overall length (RFC 5321: 254 chars max)
    if len(email) > 254:
        raise ValueError(
            f"email exceeds maximum length of 254 characters (got {len(email)})"
        )

    # Check for @ symbol
    if "@" not in email:
        raise ValueError(f"invalid email format: missing '@' symbol: '{email}'")

    # Split and validate structure
    parts = email.rsplit("@", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid email format: multiple '@' symbols: '{email}'")

    local_part, domain = parts

    # Validate local part
    if not local_part:
        raise ValueError(f"invalid email format: missing local part: '{email}'")
    if len(local_part) > 64:
        raise ValueError(
            f"email local part exceeds maximum length of 64 characters (got {len(local_part)})"
        )

    # Validate domain
    if not domain:
        raise ValueError(f"invalid email format: missing domain: '{email}'")
    if domain.startswith(".") or domain.endswith("."):
        raise ValueError(
            f"invalid email format: domain cannot start or end with '.': '{email}'"
        )
    if ".." in domain:
        raise ValueError(
            f"invalid email format: domain cannot contain consecutive dots: '{email}'"
        )
    if "." not in domain:
        raise ValueError(
            f"invalid email format: domain must contain at least one dot: '{email}'"
        )

    # Regex pattern for detailed validation (case-insensitive)
    email_pattern = r"^[a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    # Special case: single character local part
    if len(local_part) == 1:
        email_pattern = r"^[a-zA-Z0-9]@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$"

    if not re.match(email_pattern, email):
        raise ValueError(f"invalid email format: '{email}'")

    # Normalize to lowercase if requested
    if lowercase:
        email = email.lower()

    return email


def validate_ip_address(
    ip: str,
    *,
    strip: bool = True,
    version: Literal[4, 6, "any"] = "any",
    leading_zeros: bool = False,
) -> str:
    """
    Validate IP address format for IPv4 and/or IPv6.

    This function uses Python's `ipaddress` module for standards-compliant
    validation per RFC 791 (IPv4) and RFC 4291 (IPv6).

    Args:
        ip: IP address string to validate.
        strip: If True, strip leading/trailing whitespace before validation.
            Defaults to True.
        version: Required IP version. Use 4 for IPv4 only, 6 for IPv6 only,
            or "any" to accept either. Defaults to "any".
        leading_zeros: If True, allow leading zeros in IPv4 octets
            (e.g., '192.168.001.001'). Defaults to False as leading zeros
            can be ambiguous (octal vs decimal). This parameter only affects
            IPv4; IPv6 leading zeros are always allowed as they are standard
            in hexadecimal notation.

    Returns:
        The validated IP address string. If strip=True, returns the stripped
        version; otherwise returns the original input if valid. The returned
        format is normalized (IPv6 may be compressed per RFC 5952).

    Raises:
        TypeError: If ip is not a string or version is invalid.
        ValueError: If ip is empty (after stripping if enabled), has invalid
            format, or doesn't match the required version.

    Examples:
        IPv4 validation:
        >>> validate_ip_address("192.168.1.1")
        '192.168.1.1'
        >>> validate_ip_address("  10.0.0.1  ")
        '10.0.0.1'
        >>> validate_ip_address("255.255.255.255")
        '255.255.255.255'
        >>> validate_ip_address("0.0.0.0")
        '0.0.0.0'

        IPv6 validation (leading zeros always allowed):
        >>> validate_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        '2001:db8:85a3::8a2e:370:7334'
        >>> validate_ip_address("::1")
        '::1'
        >>> validate_ip_address("fe80::1")
        'fe80::1'
        >>> validate_ip_address("2001:0db8::0001")
        '2001:db8::1'

        Raises errors:
        >>> validate_ip_address("192.168.1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: '192.168.1'
        >>> validate_ip_address("not.an.ip.address")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'not.an.ip.address'
        >>> validate_ip_address("gggg::1")
        Traceback (most recent call last):
            ...
        ValueError: invalid IP address format: 'gggg::1'
    """
    # Type check
    if not isinstance(ip, str):
        raise TypeError(f"IP address must be a string, got {fmt_type(ip)}")

    if version not in (4, 6, "any"):
        raise TypeError(f"version must be 4, 6, or 'any', got {fmt_value(version)}")

    # Handle whitespace
    if strip:
        ip = ip.strip()
    elif ip != ip.strip():
        raise ValueError(
            f"invalid IP address format: contains leading or trailing whitespace: '{ip}'"
        )

    # Check for empty
    if not ip:
        raise ValueError("IP address cannot be empty")

    # Handle leading zeros in IPv4
    # IPv6 leading zeros are standard in hex notation, so no processing needed
    if "." in ip and ":" not in ip:
        # Likely IPv4
        parts = ip.split(".")
        if len(parts) == 4:
            # Check if any part has leading zeros
            has_leading_zeros = any(len(part) > 1 and part[0] == "0" for part in parts)

            if has_leading_zeros:
                if not leading_zeros:
                    # Reject leading zeros
                    raise ValueError(
                        f"invalid IPv4 address: leading zeros not allowed (octal ambiguity): '{ip}'"
                    )
                else:
                    # Strip leading zeros from each octet
                    # Convert each part to int and back to str to remove leading zeros
                    try:
                        normalized_parts = [str(int(part)) for part in parts]
                        ip = ".".join(normalized_parts)
                    except ValueError:
                        # If int() fails, it's not a valid number - let ipaddress handle it
                        pass

    # Validate using ipaddress module
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        raise ValueError(f"invalid IP address format: '{ip}'")

    # Check version requirement
    if version != "any":
        if ip_obj.version != version:
            actual_version = "IPv4" if ip_obj.version == 4 else "IPv6"
            expected_version = "IPv4" if version == 4 else "IPv6"
            raise ValueError(
                f"IP address version mismatch: '{ip}' is {actual_version}, expected {expected_version}"
            )

    # Return normalized format (ipaddress module normalizes both IPv4 and IPv6)
    return str(ip_obj)


def validate_language_code(
    language_code: str,
    allow_iso639_1: bool = True,
    allow_bcp47: bool = True,
    bcp47_parts: Literal[
        "language-region", "language-script", "language-script-region"
    ] = "language-region",
    strict: bool = True,
    case_sensitive: bool = False,
) -> str:
    """
    Validate language code against ISO 639-1 and/or BCP 47 formats.

    Args:
        language_code: The language code to validate. Leading/trailing whitespace is stripped.
        allow_iso639_1: Whether to accept ISO 639-1 two-letter codes (e.g., 'en', 'fr').
        allow_bcp47: Whether to accept BCP 47 language tags (e.g., 'en-US', 'zh-Hans-CN').
        bcp47_parts: Which BCP 47 components to allow:
            - "language-region": language-region only (e.g., 'en-US')
            - "language-script": language-script only (e.g., 'zh-Hans')
            - "language-script-region": full format (e.g., 'zh-Hans-CN')
        strict: If True, validate against official registries. If False, only validate format.
        case_sensitive: Whether validation should be case-sensitive. If False, returns lowercase.

    Returns:
        The validated language code. Normalized to lowercase unless case_sensitive=True.

    Raises:
        TypeError: If language_code is not a string.
        ValueError: If language_code is invalid, empty, or format not allowed.

    Examples:
        >>> validate_language_code('en')
        'en'
        >>> validate_language_code('EN-US')
        'en-us'
        >>> validate_language_code('  zh-Hans-CN  ')
        'zh-hans-cn'
        >>> validate_language_code('xx', strict=False)
        'xx'
        >>> validate_language_code('xx-YY', strict=False)
        'xx-yy'
    """
    # Type validation
    if not isinstance(language_code, str):
        raise TypeError(f"language code must be str, got {fmt_type(language_code)}")

    language_code = language_code.strip()

    if not language_code:
        raise ValueError("language code cannot be empty or whitespace-only")

    # At least one format must be allowed
    if not allow_iso639_1 and not allow_bcp47:
        raise ValueError(
            "at least one format (allow_iso639_1 or allow_bcp47) must be enabled"
        )

    # Normalize case for validation
    lang_lower = language_code.lower()

    # Check ISO 639-1 format
    if re.match(r"^[a-z]{2}$", lang_lower):
        if not allow_iso639_1:
            raise ValueError(
                f"ISO 639-1 format not allowed: '{language_code}'. "
                f"Use BCP 47 format (e.g., 'en-US') instead."
            )

        if strict and lang_lower not in LanguageCodes.ISO_639_1_CODES:
            raise ValueError(f"unknown ISO 639-1 language code: '{language_code}'")

        return lang_lower if not case_sensitive else language_code.strip()

    # Check BCP 47 format
    if "-" in lang_lower:
        if not allow_bcp47:
            raise ValueError(
                f"BCP 47 format not allowed: '{language_code}'. "
                f"Use ISO 639-1 format (e.g., 'en') instead."
            )

        parts = lang_lower.split("-")

        # Validate based on bcp47_parts setting
        if bcp47_parts == "language-region":
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid BCP 47 format: '{language_code}'. Expected language-region format (e.g., 'en-US')"
                )

            lang_part, region_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{2}$", region_part):
                raise ValueError(f"invalid region part in BCP 47 code: '{region_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if region_part not in CountryCodes.ISO_3166_1_CODES:
                    raise ValueError(f"unknown region code in BCP 47: '{region_part}'")

        elif bcp47_parts == "language-script":
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid BCP 47 format: '{language_code}'. Expected language-script format (e.g., 'zh-Hans')"
                )

            lang_part, script_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{4}$", script_part):
                raise ValueError(f"invalid script part in BCP 47 code: '{script_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if script_part not in LanguageCodes.ISO_15924_CODES:
                    raise ValueError(f"unknown script code in BCP 47: '{script_part}'")

        elif bcp47_parts == "language-script-region":
            if len(parts) != 3:
                raise ValueError(
                    f"invalid BCP 47 format: '{language_code}'. Expected language-script-region format (e.g., 'zh-Hans-CN')"
                )

            lang_part, script_part, region_part = parts

            # Validate format
            if not re.match(r"^[a-z]{2,3}$", lang_part):
                raise ValueError(f"invalid language part in BCP 47 code: '{lang_part}'")
            if not re.match(r"^[a-z]{4}$", script_part):
                raise ValueError(f"invalid script part in BCP 47 code: '{script_part}'")
            if not re.match(r"^[a-z]{2}$", region_part):
                raise ValueError(f"invalid region part in BCP 47 code: '{region_part}'")

            # Strict validation
            if strict:
                if lang_part not in LanguageCodes.ISO_639_1_CODES:
                    raise ValueError(f"unknown language code in BCP 47: '{lang_part}'")
                if script_part not in LanguageCodes.ISO_15924_CODES:
                    raise ValueError(f"unknown script code in BCP 47: '{script_part}'")
                if region_part not in CountryCodes.ISO_3166_1_CODES:
                    raise ValueError(f"unknown region code in BCP 47: '{region_part}'")

        return lang_lower if not case_sensitive else language_code.strip()

    # Invalid format
    allowed_formats = []
    if allow_iso639_1:
        allowed_formats.append("ISO 639-1 format (e.g., 'en', 'fr')")
    if allow_bcp47:
        if bcp47_parts == "language-region":
            allowed_formats.append(
                "BCP 47 language-region format (e.g., 'en-US', 'fr-CA')"
            )
        elif bcp47_parts == "language-script":
            allowed_formats.append(
                "BCP 47 language-script format (e.g., 'zh-Hans', 'sr-Cyrl')"
            )
        elif bcp47_parts == "language-script-region":
            allowed_formats.append(
                "BCP 47 language-script-region format (e.g., 'zh-Hans-CN', 'sr-Cyrl-RS')"
            )

    formats_str = " or ".join(allowed_formats)
    raise ValueError(
        f"invalid language code: '{language_code}'. Expected {formats_str}"
    )


def validate_url(url: str) -> str:
    """Validate URL format

    Returns:
        str: the original URL if valid

    Examples:
        >>> url = "https://www.example.com"
        >>> validate_url(url)
        'https://www.example.com'

        >>> url = "not-a-url"
        >>> validate_url(url)
        Traceback (most recent call last):
        ...
        ValueError: Invalid URL format: 'not-a-url'
    """
    if not url:
        raise ValueError("URL cannot be empty")

    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    if not re.match(url_pattern, url):
        raise ValueError(f"Invalid URL format: '{url}'")

    return url


def validate_range(
    value: float, min_value: float | None = None, max_value: float | None = None
) -> float:
    """
    Validate numeric value falls within specified bounds.

    Checks that a numeric value is within the specified minimum and maximum bounds (inclusive).
    Useful for validating ML features, hyperparameters, or any numeric data with known constraints.
    At least one bound (min or max) should be specified.

    Args:
        value: The numeric value to validate
        min_value: Minimum allowed value (inclusive), or None for no lower bound
        max_value: Maximum allowed value (inclusive), or None for no upper bound

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is outside the specified range
    """


def validate_probability(value: float) -> float:
    """
    Validate value is a valid probability between 0 and 1 (inclusive).

    Ensures a numeric value falls within the [0, 1] interval, which is required for probabilities,
    confidence scores, model outputs from sigmoid/softmax layers, mixing weights, and similar
    ML/statistical values. This is a specialized case of range validation optimized for the
    common probability constraint.

    Args:
        value: The numeric value to validate as a probability

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value is not in [0, 1]
    """


def validate_positive(value: float, strict: bool = True) -> float:
    """
    Validate value is positive (optionally allowing zero).

    Checks that a numeric value is greater than zero (or greater than/equal to zero if strict=False).
    Common for counts, distances, rates, durations, and other naturally positive quantities in
    ML pipelines. Use strict=False when zero is a valid value (e.g., counts can be zero).

    Args:
        value: The numeric value to validate
        strict: If True, value must be > 0; if False, value must be >= 0

    Returns:
        float: The original value if valid

    Raises:
        ValueError: If value violates the positivity constraint
    """


# Data Structure Validators


def validate_not_empty(collection) -> any:
    """
    Validate collection is not empty.

    Ensures a collection (list, tuple, set, dict, array, DataFrame, etc.) contains at least one
    element. Prevents silent failures from empty batches, missing data, or incorrectly filtered
    datasets that would cause downstream errors in ML pipelines or data processing.

    Args:
        collection: Any collection type that supports len() or has a boolean context

    Returns:
        The original collection if not empty

    Raises:
        ValueError: If collection is empty
    """


def validate_shape(array, expected_shape: tuple[int | None, ...]) -> any:
    """
    Validate array has expected shape/dimensions.

    Checks that an array-like object (numpy array, tensor, nested list) matches the expected shape.
    Use None in expected_shape for dimensions that can be any size (e.g., (None, 3) accepts any
    number of rows with 3 columns). Essential for validating inputs/outputs in neural networks
    and matrix operations.

    Args:
        array: Array-like object with a .shape attribute or nested structure
        expected_shape: Tuple of expected dimensions, use None for flexible dimensions

    Returns:
        The original array if shape matches

    Raises:
        ValueError: If shape doesn't match expected_shape
    """


def validate_unique(collection, key=None) -> any:
    """
    Validate all elements in collection are unique.

    Ensures no duplicate values exist in a collection, which is critical for unique identifiers,
    primary keys, index values, or when building lookup structures. For collections of objects,
    use the key parameter to specify which attribute should be unique (similar to sorted/max).

    Args:
        collection: Iterable collection to check for uniqueness
        key: Optional function to extract comparison value from each element

    Returns:
        The original collection if all elements are unique

    Raises:
        ValueError: If duplicate elements are found
    """


def validate_categorical(value: str, allowed_values: set[str] | list[str]) -> str:
    """
    Validate value is in the set of allowed categorical values.

    Checks that a value belongs to a predefined set of allowed values, which is essential for
    validating categorical features, enum-like fields, or any constrained choice fields in ML
    pipelines. Catches typos, data corruption, or schema violations early before they propagate
    through the pipeline.

    Args:
        value: The value to validate
        allowed_values: Set or list of permitted values

    Returns:
        str: The original value if it's in allowed_values

    Raises:
        ValueError: If value is not in allowed_values
    """


def validate_timestamp(timestamp_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Validate timestamp string matches expected format and is parseable.

    Checks that a timestamp string conforms to the specified format using strftime/strptime
    conventions and represents a valid datetime. Handles full date+time validation in one step,
    which is the most common temporal data format in ML pipelines (logs, events, time-series).
    Supports ISO8601, custom formats, and can validate Unix timestamps with format="unix".

    Args:
        timestamp_str: The timestamp string to validate
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")
                Use "unix" for Unix timestamp (seconds since epoch)

    Returns:
        str: The original timestamp string if valid and parseable

    Raises:
        ValueError: If timestamp doesn't match format or represents invalid datetime
    """


def validate_timestamp_range(
    timestamp_str: str,
    start: str | None = None,
    end: str | None = None,
    format: str = "%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Validate timestamp falls within specified datetime range.

    Checks that a timestamp (after parsing) falls between start and end datetimes (inclusive).
    Essential for validating temporal boundaries in time-series data, filtering event logs,
    or ensuring train/test temporal splits. All timestamps must use the same format string.

    Args:
        timestamp_str: The timestamp string to validate
        start: Minimum allowed timestamp (inclusive), or None for no lower bound
        end: Maximum allowed timestamp (inclusive), or None for no upper bound
        format: strftime/strptime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        str: The original timestamp string if within range

    Raises:
        ValueError: If timestamp is outside range or parsing fails
    """
