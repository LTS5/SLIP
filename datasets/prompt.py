
PROMPTS = {
        'PatchGastricADC22': {

            "templates":
            [
                'whole slide image showing {}'
            ],

            "slide_classnames" :

            [
                ['well differentiated tubular adenocarcinoma'],
                ['moderately differentiated tubular adenocarcinoma'],
                ['poorly differentiated adenocarcinoma'],
            ],

            "tissue_classnames" :[
                ["Normal Gastric Mucosa", "Healthy gastric lining composed of surface epithelial cells, mucous cells, chief cells, and parietal cells"],

                ["Gastric Adenocarcinoma", "Cancerous epithelial cells forming glands or tubular structures within the gastric mucosa"],

                ["Inflammatory Infiltrate", "Presence of inflammatory cells (such as lymphocytes, plasma cells, and macrophages) within the tumor stroma or surrounding tissue"],

                ["Desmoplastic Stroma", "Dense fibrous tissue surrounding the tumor cells, often indicating an aggressive tumor phenotype"],

                ["Necrotic Areas", "Regions of tissue death within the tumor mass, which can indicate rapid tumor growth or inadequate blood supply"],

                ["Ulceration", "Loss of epithelial lining with exposure of the underlying tissue, often seen in more advanced stages of adenocarcinoma"],

                ["Lymphovascular Invasion", "Invasion of cancer cells into lymphatic or blood vessels, which may appear as tumor cells within vascular lumens"],

                ["Perineural Invasion", "Invasion of cancer cells into nerves, typically observed as tumor cells surrounding nerve bundles"],

                ["Metastatic Deposits", "Presence of tumor cells in regional lymph nodes or distant organs, indicating advanced disease stage"],

                ["Focal Dysplasia", "Areas of abnormal cellular growth and differentiation, which may precede the development of adenocarcinoma"],

                ["Surface Epithelial Changes", "Alterations in the surface epithelium such as hyperplasia, dysplasia, or metaplasia, which can be precursors to adenocarcinoma"],

                ["Submucosal Invasion", "Invasion of tumor cells beyond the mucosal layer into the submucosa of the stomach wall"],

                ["Muscularis Propria Invasion", "Invasion of tumor cells into the muscular layer of the stomach wall"],

                ["Lymph Node Architecture", "Examination of lymph node architecture for evidence of tumor involvement, such as effacement of normal nodal architecture by tumor cells"],

                ["Lymph Node Metastasis", "Presence of tumor cells within lymph node sinuses or parenchyma, indicating spread from the primary tumor"],

                ["Intramucosal Carcinoma", "Adenocarcinoma confined to the mucosal layer without invasion into deeper layers of the stomach wall"],

                ["Subserosal Invasion", "Invasion of tumor cells into the subserosal layer of the stomach wall, beyond the serosa"],

                ["Serosal Involvement", "Invasion of tumor cells through the serosal layer, potentially leading to peritoneal spread"],
            ]
    },

    "DHMC" : {

        "templates":
            [
                '{}'
            ],

        "slide_classnames":
            [
                ["lepidic pattern adenocarcinoma",],
                ["acinar pattern adenocarcinoma",],
                ["solid pattern adenocarcinoma",]
            ],

        "tissue_classnames": [
                ["Alveolar Structure", "Evaluation of alveolar architecture, including alveolar septa thickness, presence of alveolar hemorrhage, or edema"],
                ["Bronchiolar Architecture", "Assessment of bronchiolar structures for abnormalities such as inflammation, metaplasia, or fibrosis"],
                ["Interstitial Tissue", "Examination of interstitial tissue for signs of fibrosis, inflammation, or interstitial lung diseases"],
                ["Blood Vessels", "Evaluation of blood vessels for signs of vasculitis, thrombosis, or hemorrhage"],
                ["Inflammatory Infiltrates", "Identification of inflammatory cells such as neutrophils, lymphocytes, eosinophils, or macrophages within the lung parenchyma"],
                ["Necrosis", "Presence of necrotic tissue, which may indicate acute injury or infection"],
                ["Fibrosis", "Detection and grading of fibrotic changes within the lung parenchyma, including patterns such as organizing pneumonia, usual interstitial pneumonia, or nonspecific interstitial pneumonia"],
                ["Pneumocytes", "Assessment of type I and type II pneumocytes for signs of injury or hyperplasia"],
                ["Pleura", "Examination of the pleural surface for abnormalities such as thickening, fibrosis, or the presence of tumor cells"],
                ["Inclusions or Deposits", "Identification of abnormal inclusions or deposits within the lung tissue, such as amyloidosis, hemosiderosis, or foreign body granulomas"],
                ["Microorganisms", "Detection of microorganisms such as bacteria, viruses, fungi, or parasites, which may indicate infectious processes"],
                ["Metaplasia or Dysplasia", "Evaluation of epithelial cells for metaplastic or dysplastic changes, which may indicate premalignant or malignant conditions"],
                ["Tumor Morphology", "Assessment of tumor characteristics including size, shape, margins, and differentiation"],
                ["Calcifications", "Identification of calcified structures within the lung tissue, which may indicate chronic inflammatory processes or neoplastic conditions"],
                ["Artifacts", "Recognition and differentiation of artifacts from true pathological changes to ensure accurate interpretation of tissue characteristics"]
        ]
    },

    "TCGA" : {

        "templates":
            [
                '{}'
            ],

        "slide_classnames":
            [
                ["lung adenocarcinoma",],
                ["lung squamous cell carcinoma",],
            ],

        "tissue_classnames": [
                ["Alveolar tissue", "The normal lung tissue, consisting of air spaces and alveolar sacs. Typically seen in adenocarcinoma cases."],
                ["Tumor nests", "Clusters of cancerous cells that form distinct groups, often more pronounced in squamous cell carcinoma."],
                ["Solid tumor", "A solid mass of cancer cells, more frequent in adenocarcinoma but also found in squamous cell carcinoma."],
                ["Keratin pearls", "Concentric layers of keratin seen in squamous cell carcinoma, an important diagnostic feature."],
                ["Glandular formation", "Structures resembling glands, indicative of adenocarcinoma, formed by malignant cells."],
                ["Squamous differentiation", "A hallmark of squamous cell carcinoma, including keratinization, intercellular bridges, and squamous cells."],
                ["Stroma", "The supportive tissue around the tumor, providing context to cancer growth. Can vary between LADC and LSCC."],
                ["Necrosis", "Dead or dying tumor tissue, commonly seen in both types but may exhibit different patterns."],
                ["Mucin production", "Presence of mucin (slimy protein substance) in cancer cells, a characteristic of adenocarcinoma."],
                ["Basement membrane", "A thin layer that separates epithelial tissue from underlying connective tissue, can be disrupted by tumors."],
                ["Inflammatory infiltrate", "Immune cells that have infiltrated the tumor, varying in quantity and type between the two carcinoma types."],
                ["Vascular invasion", "Tumor cells invading blood vessels, a marker of aggressiveness, often assessed in both cancer types."],
                ["Fibrosis", "Scarring tissue, can be associated with tumor growth, commonly seen in adenocarcinoma."],
                ["Pleomorphism", "Variability in the size and shape of cancer cells, observed in both adenocarcinoma and squamous cell carcinoma."],
                ["Lymphocytic infiltration", "The presence of lymphocytes, which may vary between tumor types and can help distinguish between them."],
                ["Bronchial epithelium", "Lining of the bronchial tubes, often shows changes in squamous cell carcinoma cases."],
                ["Hyalinization", "Increased deposition of extracellular matrix proteins, often found in squamous cell carcinoma stroma."]
        ]
    }
}