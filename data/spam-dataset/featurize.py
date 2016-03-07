# -*- coding: utf-8 -*-
'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
	-'training_data'
	-'training_labels'
	-'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
	return float(freq['pain'])

def freq_private_feature(text, freq):
	return float(freq['private'])

def freq_bank_feature(text, freq):
	return float(freq['bank'])

def freq_money_feature(text, freq):
	return float(freq['money'])

def freq_drug_feature(text, freq):
	return float(freq['drug'])

def freq_spam_feature(text, freq):
	return float(freq['spam'])

def freq_prescription_feature(text, freq):
	return float(freq['prescription'])

def freq_creative_feature(text, freq):
	return float(freq['creative'])

def freq_height_feature(text, freq):
	return float(freq['height'])

def freq_featured_feature(text, freq):
	return float(freq['featured'])

def freq_differ_feature(text, freq):
	return float(freq['differ'])

def freq_energy_feature(text, freq):
	return float(freq['energy'])

def freq_message_feature(text, freq):
	return float(freq['message'])

def freq_volumes_feature(text, freq):
	return float(freq['volumes'])

def freq_revision_feature(text, freq):
	return float(freq['revision'])

def freq_memo_feature(text, freq):
	return float(freq['memo'])

def freq_planning_feature(text, freq):
	return float(freq['planning'])

def freq_out_feature(text, freq):
	return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
	return text.count(';')

def freq_dollar_feature(text, freq):
	return text.count('$')

def freq_sharp_feature(text, freq):
	return text.count('#')

def freq_exclamation_feature(text, freq):
	return text.count('!')

def freq_para_feature(text, freq):
	return text.count('(')

def freq_and_feature(text, freq):
	return text.count('&')

def freq_colon_feature(text, freq):
	return text.count(':')

def freq_double_quotes_feature(text, freq):
	return text.count('"')

def freq_single_quote_feature(text, freq):
	return text.count('\'')

def freq_front_slash_feature(text, freq):
	return text.count('/')

def freq_closing_paren_feature(text, freq):
	return text.count(')')

def freq_pipe_feature(text, freq):
	return text.count('|')

def freq_dash_feature(text, freq):
	return text.count('-')

def freq_caret_feature(text, freq):
	return text.count('^')

def freq_at_feature(text, freq):
	return text.count('@')

def freq_greater_than_feature(text, freq):
	return text.count('>')

# Misc features
def freq_digit_feature(text, freq):
	return sum(c.isdigit() for c in text)

def freq_online_feature(text, freq):
	return float(freq['online'])

def freq_enlarge_feature(text, freq):
	return float(freq['enlarge'])

def freq_click_feature(text, freq):
	return float(freq['click'])

def freq_here_feature(text, freq):
	return float(freq['here'])

def freq_guaranteed_feature(text, freq):
	return float(freq['guaranteed'])

def freq_free_feature(text, freq):
	return float(freq['free'])

def freq_please_feature(text, freq):
	return float(freq['please'])

# --------- Spam features ----------

def freq_td_feature(text, freq):
	return float(freq['td'])

def freq_nbsp_feature(text, freq):
	return float(freq['nbsp'])

def freq_pills_feature(text, freq):
	return float(freq['pills'])

def freq_width_feature(text, freq):
	return float(freq['width'])

def freq_computron_feature(text, freq):
	return float(freq['computron'])

def freq_href_feature(text, freq):
	return float(freq['href'])

def freq_viagra_feature(text, freq):
	return float(freq['viagra'])

def freq_xp_feature(text, freq):
	return float(freq['xp'])

def freq_src_feature(text, freq):
	return float(freq['src'])

def freq_cialis_feature(text, freq):
	return float(freq['cialis'])

def freq_soft_feature(text, freq):
	return float(freq['soft'])

def freq_meds_feature(text, freq):
	return float(freq['meds'])

def freq_paliourg_feature(text, freq):
	return float(freq['paliourg'])

def freq_voip_feature(text, freq):
	return float(freq['voip'])

def freq_oo_feature(text, freq):
	return float(freq['oo'])

def freq_php_feature(text, freq):
	return float(freq['php'])

def freq_bgcolor_feature(text, freq):
	return float(freq['bgcolor'])

def freq_drugs_feature(text, freq):
	return float(freq['drugs'])

def freq_biz_feature(text, freq):
	return float(freq['biz'])

def freq_mx_feature(text, freq):
	return float(freq['mx'])

def freq_img_feature(text, freq):
	return float(freq['img'])

def freq_photoshop_feature(text, freq):
	return float(freq['photoshop'])

def freq_valign_feature(text, freq):
	return float(freq['valign'])

def freq_uncertainties_feature(text, freq):
	return float(freq['uncertainties'])

def freq_000000_feature(text, freq):
	return float(freq['000000'])

def freq_div_feature(text, freq):
	return float(freq['div'])

def freq_hotlist_feature(text, freq):
	return float(freq['hotlist'])

def freq_moopid_feature(text, freq):
	return float(freq['moopid'])

def freq_pharmacy_feature(text, freq):
	return float(freq['pharmacy'])

def freq_projections_feature(text, freq):
	return float(freq['projections'])

def freq_gr_feature(text, freq):
	return float(freq['gr'])

def freq_xanax_feature(text, freq):
	return float(freq['xanax'])

def freq_htmlimg_feature(text, freq):
	return float(freq['htmlimg'])

def freq_colspan_feature(text, freq):
	return float(freq['colspan'])

def freq_corel_feature(text, freq):
	return float(freq['corel'])

def freq_readers_feature(text, freq):
	return float(freq['readers'])

def freq_macromedia_feature(text, freq):
	return float(freq['macromedia'])

def freq_dealer_feature(text, freq):
	return float(freq['dealer'])

def freq_knle_feature(text, freq):
	return float(freq['knle'])

def freq_valium_feature(text, freq):
	return float(freq['valium'])

def freq_demokritos_feature(text, freq):
	return float(freq['demokritos'])

def freq_iit_feature(text, freq):
	return float(freq['iit'])

def freq_rolex_feature(text, freq):
	return float(freq['rolex'])

def freq_intel_feature(text, freq):
	return float(freq['intel'])

def freq_speculative_feature(text, freq):
	return float(freq['speculative'])

def freq_1933_feature(text, freq):
	return float(freq['1933'])

def freq_rnd_feature(text, freq):
	return float(freq['rnd'])

def freq_rx_feature(text, freq):
	return float(freq['rx'])

def freq_jebel_feature(text, freq):
	return float(freq['jebel'])

def freq_lots_feature(text, freq):
	return float(freq['lots'])

def freq_half_feature(text, freq):
	return float(freq['½'])

def freq_alt_feature(text, freq):
	return float(freq['alt'])

def freq_ooking_feature(text, freq):
	return float(freq['ooking'])

def freq_darial_feature(text, freq):
	return float(freq['darial'])

def freq_yap_feature(text, freq):
	return float(freq['yap'])

def freq_gra_feature(text, freq):
	return float(freq['gra'])

def freq_0310041_feature(text, freq):
	return float(freq['0310041'])

def freq_canon_feature(text, freq):
	return float(freq['canon'])

def freq_export_feature(text, freq):
	return float(freq['export'])

def freq_1934_feature(text, freq):
	return float(freq['1934'])

def freq_0424040_feature(text, freq):
	return float(freq['0424040'])

def freq_pill_feature(text, freq):
	return float(freq['pill'])

def freq_ftar_feature(text, freq):
	return float(freq['ftar'])

def freq_logos_feature(text, freq):
	return float(freq['logos'])

def freq_toshiba_feature(text, freq):
	return float(freq['toshiba'])

def freq_soma_feature(text, freq):
	return float(freq['soma'])

def freq_fontfont_feature(text, freq):
	return float(freq['fontfont'])

def freq_materially_feature(text, freq):
	return float(freq['materially'])

def freq_vicodin_feature(text, freq):
	return float(freq['vicodin'])

def freq_resellers_feature(text, freq):
	return float(freq['resellers'])

def freq_viewsonic_feature(text, freq):
	return float(freq['viewsonic'])

def freq_cellpadding_feature(text, freq):
	return float(freq['cellpadding'])

def freq_1618_feature(text, freq):
	return float(freq['1618'])

def freq_robotics_feature(text, freq):
	return float(freq['robotics'])

def freq_8834464_feature(text, freq):
	return float(freq['8834464'])

def freq_hewlett_feature(text, freq):
	return float(freq['hewlett'])

def freq_8834454_feature(text, freq):
	return float(freq['8834454'])

def freq_4176_feature(text, freq):
	return float(freq['4176'])

def freq_materia_feature(text, freq):
	return float(freq['materia'])

def freq_1226030_feature(text, freq):
	return float(freq['1226030'])

def freq_apc_feature(text, freq):
	return float(freq['apc'])

def freq_iomega_feature(text, freq):
	return float(freq['iomega'])

def freq_targus_feature(text, freq):
	return float(freq['targus'])

def freq_aopen_feature(text, freq):
	return float(freq['aopen'])

def freq_customerservice_feature(text, freq):
	return float(freq['customerservice'])

def freq_nomad_feature(text, freq):
	return float(freq['nomad'])

def freq_packard_feature(text, freq):
	return float(freq['packard'])

def freq_intellinet_feature(text, freq):
	return float(freq['intellinet'])

def freq_enquiries_feature(text, freq):
	return float(freq['enquiries'])

def freq_construed_feature(text, freq):
	return float(freq['construed'])

def freq_drug_feature(text, freq):
	return float(freq['drug'])

def freq_uae_feature(text, freq):
	return float(freq['uae'])

def freq_itoy_feature(text, freq):
	return float(freq['itoy'])

def freq_predictions_feature(text, freq):
	return float(freq['predictions'])

def freq_muscle_feature(text, freq):
	return float(freq['muscle'])

def freq_zonedubai_feature(text, freq):
	return float(freq['zonedubai'])

def freq_emirates_feature(text, freq):
	return float(freq['emirates'])

def freq_leth_feature(text, freq):
	return float(freq['leth'])

def freq_anticipates_feature(text, freq):
	return float(freq['anticipates'])

def freq_aeor_feature(text, freq):
	return float(freq['aeor'])

def freq_oem_feature(text, freq):
	return float(freq['oem'])

def freq_otcbb_feature(text, freq):
	return float(freq['otcbb'])

def freq_assurance_feature(text, freq):
	return float(freq['assurance'])

def freq_internationa_feature(text, freq):
	return float(freq['internationa'])

def freq_female_feature(text, freq):
	return float(freq['female'])

def freq_resuits_feature(text, freq):
	return float(freq['resuits'])

def freq_studio_feature(text, freq):
	return float(freq['studio'])

def freq_spain_feature(text, freq):
	return float(freq['spain'])

def freq_palestinian_feature(text, freq):
	return float(freq['palestinian'])

def freq_illustrator_feature(text, freq):
	return float(freq['illustrator'])

def freq_verdana_feature(text, freq):
	return float(freq['verdana'])

def freq_aerofoam_feature(text, freq):
	return float(freq['aerofoam'])

def freq_mining_feature(text, freq):
	return float(freq['mining'])

def freq_arial_feature(text, freq):
	return float(freq['arial'])

def freq_graphics_feature(text, freq):
	return float(freq['graphics'])

def freq_vi_feature(text, freq):
	return float(freq['vi'])

def freq_abdv_feature(text, freq):
	return float(freq['abdv'])

def freq_wysak_feature(text, freq):
	return float(freq['wysak'])

def freq_sofftwaares_feature(text, freq):
	return float(freq['sofftwaares'])

def freq_apple_feature(text, freq):
	return float(freq['apple'])

def freq_eogi_feature(text, freq):
	return float(freq['eogi'])

def freq_erections_feature(text, freq):
	return float(freq['erections'])

def freq_three_feature(text, freq):
	return float(freq['³'])

def freq_pubiisher_feature(text, freq):
	return float(freq['pubiisher'])

def freq_wiil_feature(text, freq):
	return float(freq['wiil'])

def freq_gains_feature(text, freq):
	return float(freq['gains'])

def freq_ffffff_feature(text, freq):
	return float(freq['ffffff'])

def freq_phentermine_feature(text, freq):
	return float(freq['phentermine'])

def freq_hottlist_feature(text, freq):
	return float(freq['hottlist'])

def freq_cellspacing_feature(text, freq):
	return float(freq['cellspacing'])

def freq_male_feature(text, freq):
	return float(freq['male'])

def freq_atleast_feature(text, freq):
	return float(freq['atleast'])

def freq_dosage_feature(text, freq):
	return float(freq['dosage'])

def freq_netherlands_feature(text, freq):
	return float(freq['netherlands'])

def freq_ail_feature(text, freq):
	return float(freq['ail'])

def freq_prescriptions_feature(text, freq):
	return float(freq['prescriptions'])

def freq_undervalued_feature(text, freq):
	return float(freq['undervalued'])

def freq_cia_feature(text, freq):
	return float(freq['cia'])

def freq_couid_feature(text, freq):
	return float(freq['couid'])

def freq_technoiogies_feature(text, freq):
	return float(freq['technoiogies'])

def freq_dose_feature(text, freq):
	return float(freq['dose'])

def freq_opinions_feature(text, freq):
	return float(freq['opinions'])

def freq_sir_feature(text, freq):
	return float(freq['sir'])

def freq_ur_feature(text, freq):
	return float(freq['ur'])

def freq_kin_feature(text, freq):
	return float(freq['kin'])

def freq_discreet_feature(text, freq):
	return float(freq['discreet'])

def freq_prozac_feature(text, freq):
	return float(freq['prozac'])

def freq_brbr_feature(text, freq):
	return float(freq['brbr'])

def freq_tirr_feature(text, freq):
	return float(freq['tirr'])

def freq_serial_feature(text, freq):
	return float(freq['serial'])

def freq_tadalafil_feature(text, freq):
	return float(freq['tadalafil'])

def freq_pinnacle_feature(text, freq):
	return float(freq['pinnacle'])

def freq_distributorjebel_feature(text, freq):
	return float(freq['distributorjebel'])

def freq_mai_feature(text, freq):
	return float(freq['mai'])

def freq_artprice_feature(text, freq):
	return float(freq['artprice'])

def freq_iso_feature(text, freq):
	return float(freq['iso'])

def freq_ambien_feature(text, freq):
	return float(freq['ambien'])

def freq_cheapest_feature(text, freq):
	return float(freq['cheapest'])

def freq_vlagra_feature(text, freq):
	return float(freq['vlagra'])

def freq_lasts_feature(text, freq):
	return float(freq['lasts'])

def freq_microcap_feature(text, freq):
	return float(freq['microcap'])

def freq_techlite_feature(text, freq):
	return float(freq['techlite'])

def freq_accessories_feature(text, freq):
	return float(freq['accessories'])

def freq_indicative_feature(text, freq):
	return float(freq['indicative'])

def freq_impotence_feature(text, freq):
	return float(freq['impotence'])

def freq_levitra_feature(text, freq):
	return float(freq['levitra'])

def freq_alcohol_feature(text, freq):
	return float(freq['alcohol'])

def freq_lls_feature(text, freq):
	return float(freq['lls'])

def freq_manufacturer_feature(text, freq):
	return float(freq['manufacturer'])

def freq_nigeria_feature(text, freq):
	return float(freq['nigeria'])

def freq_imited_feature(text, freq):
	return float(freq['imited'])

def freq_pr_feature(text, freq):
	return float(freq['pr'])

def freq_emerson_feature(text, freq):
	return float(freq['emerson'])

def freq_erection_feature(text, freq):
	return float(freq['erection'])

def freq_der_feature(text, freq):
	return float(freq['der'])

def freq_es_feature(text, freq):
	return float(freq['es'])

def freq_israeli_feature(text, freq):
	return float(freq['israeli'])

def freq_constitutes_feature(text, freq):
	return float(freq['constitutes'])

def freq_ooo_feature(text, freq):
	return float(freq['ooo'])

def freq_spacer_feature(text, freq):
	return float(freq['spacer'])

def freq_piease_feature(text, freq):
	return float(freq['piease'])

def freq_porn_feature(text, freq):
	return float(freq['porn'])

def freq_bingo_feature(text, freq):
	return float(freq['bingo'])

def freq_paypal_feature(text, freq):
	return float(freq['paypal'])

def freq_reform_feature(text, freq):
	return float(freq['reform'])

def freq_verge_feature(text, freq):
	return float(freq['verge'])

def freq_vl_feature(text, freq):
	return float(freq['vl'])

def freq_fontbr_feature(text, freq):
	return float(freq['fontbr'])

def freq_ffffffstrongfont_feature(text, freq):
	return float(freq['ffffffstrongfont'])

def freq_sans_feature(text, freq):
	return float(freq['sans'])

def freq_macintosh_feature(text, freq):
	return float(freq['macintosh'])

def freq_newsietter_feature(text, freq):
	return float(freq['newsietter'])

def freq_bias_feature(text, freq):
	return float(freq['bias'])

def freq_omit_feature(text, freq):
	return float(freq['omit'])

def freq_packaging_feature(text, freq):
	return float(freq['packaging'])

def freq_replica_feature(text, freq):
	return float(freq['replica'])

def freq_genuine_feature(text, freq):
	return float(freq['genuine'])

def freq_cum_feature(text, freq):
	return float(freq['cum'])

def freq_ion_feature(text, freq):
	return float(freq['ion'])

def freq_biosphere_feature(text, freq):
	return float(freq['biosphere'])

# --------- Ham features ----------

def freq_enron_feature(text, freq):
	return float(freq['enron'])

def freq_meter_feature(text, freq):
	return float(freq['meter'])

def freq_hpl_feature(text, freq):
	return float(freq['hpl'])

def freq_daren_feature(text, freq):
	return float(freq['daren'])

def freq_mmbtu_feature(text, freq):
	return float(freq['mmbtu'])

def freq_xls_feature(text, freq):
	return float(freq['xls'])

def freq_sitara_feature(text, freq):
	return float(freq['sitara'])

def freq_volumes_feature(text, freq):
	return float(freq['volumes'])

def freq_pec_feature(text, freq):
	return float(freq['pec'])

def freq_ena_feature(text, freq):
	return float(freq['ena'])

def freq_melissa_feature(text, freq):
	return float(freq['melissa'])

def freq_teco_feature(text, freq):
	return float(freq['teco'])

def freq_tenaska_feature(text, freq):
	return float(freq['tenaska'])

def freq_pat_feature(text, freq):
	return float(freq['pat'])

def freq_aimee_feature(text, freq):
	return float(freq['aimee'])

def freq_noms_feature(text, freq):
	return float(freq['noms'])

def freq_actuals_feature(text, freq):
	return float(freq['actuals'])

def freq_hsc_feature(text, freq):
	return float(freq['hsc'])

def freq_cotten_feature(text, freq):
	return float(freq['cotten'])

def freq_chokshi_feature(text, freq):
	return float(freq['chokshi'])

def freq_fyi_feature(text, freq):
	return float(freq['fyi'])

def freq_hplc_feature(text, freq):
	return float(freq['hplc'])

def freq_wellhead_feature(text, freq):
	return float(freq['wellhead'])

def freq_clynes_feature(text, freq):
	return float(freq['clynes'])

def freq_eastrans_feature(text, freq):
	return float(freq['eastrans'])

def freq_counterparty_feature(text, freq):
	return float(freq['counterparty'])

def freq_txu_feature(text, freq):
	return float(freq['txu'])

def freq_hplno_feature(text, freq):
	return float(freq['hplno'])

def freq_rita_feature(text, freq):
	return float(freq['rita'])

def freq_lannou_feature(text, freq):
	return float(freq['lannou'])

def freq_nominations_feature(text, freq):
	return float(freq['nominations'])

def freq_pefs_feature(text, freq):
	return float(freq['pefs'])

def freq_enronxgate_feature(text, freq):
	return float(freq['enronxgate'])

def freq_weissman_feature(text, freq):
	return float(freq['weissman'])

def freq_gcs_feature(text, freq):
	return float(freq['gcs'])

def freq_cec_feature(text, freq):
	return float(freq['cec'])

def freq_wynne_feature(text, freq):
	return float(freq['wynne'])

def freq_hplo_feature(text, freq):
	return float(freq['hplo'])

def freq_allocated_feature(text, freq):
	return float(freq['allocated'])

def freq_iferc_feature(text, freq):
	return float(freq['iferc'])

def freq_path_feature(text, freq):
	return float(freq['path'])

def freq_spreadsheet_feature(text, freq):
	return float(freq['spreadsheet'])

def freq_anita_feature(text, freq):
	return float(freq['anita'])

def freq_buyback_feature(text, freq):
	return float(freq['buyback'])

def freq_equistar_feature(text, freq):
	return float(freq['equistar'])

def freq_sherlyn_feature(text, freq):
	return float(freq['sherlyn'])

def freq_flowed_feature(text, freq):
	return float(freq['flowed'])

def freq_pops_feature(text, freq):
	return float(freq['pops'])

def freq_scheduling_feature(text, freq):
	return float(freq['scheduling'])

def freq_entex_feature(text, freq):
	return float(freq['entex'])

def freq_katy_feature(text, freq):
	return float(freq['katy'])

def freq_ees_feature(text, freq):
	return float(freq['ees'])

def freq_clem_feature(text, freq):
	return float(freq['clem'])

def freq_darren_feature(text, freq):
	return float(freq['darren'])

def freq_calpine_feature(text, freq):
	return float(freq['calpine'])

def freq_gco_feature(text, freq):
	return float(freq['gco'])

def freq_aep_feature(text, freq):
	return float(freq['aep'])

def freq_midcon_feature(text, freq):
	return float(freq['midcon'])

def freq_cornhusker_feature(text, freq):
	return float(freq['cornhusker'])

def freq_redeliveries_feature(text, freq):
	return float(freq['redeliveries'])

def freq_victor_feature(text, freq):
	return float(freq['victor'])

def freq_schumack_feature(text, freq):
	return float(freq['schumack'])

def freq_reinhardt_feature(text, freq):
	return float(freq['reinhardt'])

def freq_luong_feature(text, freq):
	return float(freq['luong'])

def freq_lsk_feature(text, freq):
	return float(freq['lsk'])

def freq_herod_feature(text, freq):
	return float(freq['herod'])

def freq_hplnl_feature(text, freq):
	return float(freq['hplnl'])

def freq_methanol_feature(text, freq):
	return float(freq['methanol'])

def freq_revision_feature(text, freq):
	return float(freq['revision'])

def freq_6353_feature(text, freq):
	return float(freq['6353'])

def freq_papayoti_feature(text, freq):
	return float(freq['papayoti'])

def freq_dfarmer_feature(text, freq):
	return float(freq['dfarmer'])

def freq_bryan_feature(text, freq):
	return float(freq['bryan'])

def freq_lamphier_feature(text, freq):
	return float(freq['lamphier'])

def freq_valero_feature(text, freq):
	return float(freq['valero'])

def freq_cleburne_feature(text, freq):
	return float(freq['cleburne'])

def freq_baumbach_feature(text, freq):
	return float(freq['baumbach'])

def freq_poorman_feature(text, freq):
	return float(freq['poorman'])

def freq_liz_feature(text, freq):
	return float(freq['liz'])

def freq_eol_feature(text, freq):
	return float(freq['eol'])

def freq_lsp_feature(text, freq):
	return float(freq['lsp'])

def freq_waha_feature(text, freq):
	return float(freq['waha'])

def freq_riley_feature(text, freq):
	return float(freq['riley'])

def freq_enrononline_feature(text, freq):
	return float(freq['enrononline'])

def freq_enserch_feature(text, freq):
	return float(freq['enserch'])

def freq_outage_feature(text, freq):
	return float(freq['outage'])

def freq_employee_feature(text, freq):
	return float(freq['employee'])

def freq_098_feature(text, freq):
	return float(freq['098'])

def freq_cernosek_feature(text, freq):
	return float(freq['cernosek'])

def freq_pathed_feature(text, freq):
	return float(freq['pathed'])

def freq_853_feature(text, freq):
	return float(freq['853'])

def freq_aepin_feature(text, freq):
	return float(freq['aepin'])

def freq_tejas_feature(text, freq):
	return float(freq['tejas'])

def freq_saturday_feature(text, freq):
	return float(freq['saturday'])

def freq_withers_feature(text, freq):
	return float(freq['withers'])

def freq_boas_feature(text, freq):
	return float(freq['boas'])

def freq_superty_feature(text, freq):
	return float(freq['superty'])

def freq_avila_feature(text, freq):
	return float(freq['avila'])

def freq_easttexas_feature(text, freq):
	return float(freq['easttexas'])

def freq_hesco_feature(text, freq):
	return float(freq['hesco'])

def freq_lamadrid_feature(text, freq):
	return float(freq['lamadrid'])

def freq_cdnow_feature(text, freq):
	return float(freq['cdnow'])

def freq_herrera_feature(text, freq):
	return float(freq['herrera'])

def freq_gpgfin_feature(text, freq):
	return float(freq['gpgfin'])

def freq_hakemack_feature(text, freq):
	return float(freq['hakemack'])

def freq_sandi_feature(text, freq):
	return float(freq['sandi'])

def freq_paso_feature(text, freq):
	return float(freq['paso'])

def freq_cp_feature(text, freq):
	return float(freq['cp'])

def freq_kcs_feature(text, freq):
	return float(freq['kcs'])

def freq_eileen_feature(text, freq):
	return float(freq['eileen'])

def freq_reliantenergy_feature(text, freq):
	return float(freq['reliantenergy'])

def freq_4179_feature(text, freq):
	return float(freq['4179'])

def freq_comments_feature(text, freq):
	return float(freq['comments'])

def freq_9497_feature(text, freq):
	return float(freq['9497'])

def freq_345_feature(text, freq):
	return float(freq['345'])

def freq_gomes_feature(text, freq):
	return float(freq['gomes'])

def freq_ponton_feature(text, freq):
	return float(freq['ponton'])

def freq_enw_feature(text, freq):
	return float(freq['enw'])

def freq_1266_feature(text, freq):
	return float(freq['1266'])

def freq_enerfin_feature(text, freq):
	return float(freq['enerfin'])

def freq_neon_feature(text, freq):
	return float(freq['neon'])

def freq_tisdale_feature(text, freq):
	return float(freq['tisdale'])

def freq_dynegy_feature(text, freq):
	return float(freq['dynegy'])

def freq_strangers_feature(text, freq):
	return float(freq['strangers'])

def freq_tetco_feature(text, freq):
	return float(freq['tetco'])

def freq_mckay_feature(text, freq):
	return float(freq['mckay'])

def freq_invoices_feature(text, freq):
	return float(freq['invoices'])

def freq_neuweiler_feature(text, freq):
	return float(freq['neuweiler'])

def freq_intrastate_feature(text, freq):
	return float(freq['intrastate'])

def freq_mops_feature(text, freq):
	return float(freq['mops'])

def freq_charlene_feature(text, freq):
	return float(freq['charlene'])

def freq_trader_feature(text, freq):
	return float(freq['trader'])

def freq_interconnect_feature(text, freq):
	return float(freq['interconnect'])

def freq_0435_feature(text, freq):
	return float(freq['0435'])

def freq_gtc_feature(text, freq):
	return float(freq['gtc'])

def freq_olsen_feature(text, freq):
	return float(freq['olsen'])

def freq_availabilities_feature(text, freq):
	return float(freq['availabilities'])

def freq_tufco_feature(text, freq):
	return float(freq['tufco'])

def freq_panenergy_feature(text, freq):
	return float(freq['panenergy'])

def freq_heidi_feature(text, freq):
	return float(freq['heidi'])

def freq_valadez_feature(text, freq):
	return float(freq['valadez'])

def freq_billed_feature(text, freq):
	return float(freq['billed'])

def freq_77002_feature(text, freq):
	return float(freq['77002'])

def freq_cass_feature(text, freq):
	return float(freq['cass'])

def freq_revisions_feature(text, freq):
	return float(freq['revisions'])

def freq_unaccounted_feature(text, freq):
	return float(freq['unaccounted'])

def freq_brazos_feature(text, freq):
	return float(freq['brazos'])

def freq_copano_feature(text, freq):
	return float(freq['copano'])

def freq_origination_feature(text, freq):
	return float(freq['origination'])

def freq_winfree_feature(text, freq):
	return float(freq['winfree'])

def freq_epgt_feature(text, freq):
	return float(freq['epgt'])

def freq_christy_feature(text, freq):
	return float(freq['christy'])

def freq_reveffo_feature(text, freq):
	return float(freq['reveffo'])

def freq_crosstex_feature(text, freq):
	return float(freq['crosstex'])

def freq_troy_feature(text, freq):
	return float(freq['troy'])

def freq_dth_feature(text, freq):
	return float(freq['dth'])

def freq_lonestar_feature(text, freq):
	return float(freq['lonestar'])

def freq_rick_feature(text, freq):
	return float(freq['rick'])

def freq_encina_feature(text, freq):
	return float(freq['encina'])

def freq_kelly_feature(text, freq):
	return float(freq['kelly'])

def freq_ews_feature(text, freq):
	return float(freq['ews'])

def freq_mmbtus_feature(text, freq):
	return float(freq['mmbtus'])

def freq_cpr_feature(text, freq):
	return float(freq['cpr'])

def freq_mtr_feature(text, freq):
	return float(freq['mtr'])

def freq_csikos_feature(text, freq):
	return float(freq['csikos'])

def freq_goliad_feature(text, freq):
	return float(freq['goliad'])

def freq_pager_feature(text, freq):
	return float(freq['pager'])

def freq_assigned_feature(text, freq):
	return float(freq['assigned'])

def freq_vols_feature(text, freq):
	return float(freq['vols'])

def freq_mcmills_feature(text, freq):
	return float(freq['mcmills'])

def freq_beaty_feature(text, freq):
	return float(freq['beaty'])

def freq_marta_feature(text, freq):
	return float(freq['marta'])

def freq_kristen_feature(text, freq):
	return float(freq['kristen'])

def freq_gdp_feature(text, freq):
	return float(freq['gdp'])

def freq_gisb_feature(text, freq):
	return float(freq['gisb'])

def freq_henderson_feature(text, freq):
	return float(freq['henderson'])

def freq_463_feature(text, freq):
	return float(freq['463'])

def freq_sweeney_feature(text, freq):
	return float(freq['sweeney'])

def freq_texoma_feature(text, freq):
	return float(freq['texoma'])

def freq_hernandez_feature(text, freq):
	return float(freq['hernandez'])

def freq_ebs_feature(text, freq):
	return float(freq['ebs'])

def freq_hillary_feature(text, freq):
	return float(freq['hillary'])

def freq_seaman_feature(text, freq):
	return float(freq['seaman'])

def freq_schedulers_feature(text, freq):
	return float(freq['schedulers'])

def freq_patti_feature(text, freq):
	return float(freq['patti'])

def freq_lindley_feature(text, freq):
	return float(freq['lindley'])

def freq_memo_feature(text, freq):
	return float(freq['memo'])

def freq_mazowita_feature(text, freq):
	return float(freq['mazowita'])

def freq_pathing_feature(text, freq):
	return float(freq['pathing'])

def freq_984132_feature(text, freq):
	return float(freq['984132'])

def freq_pgev_feature(text, freq):
	return float(freq['pgev'])

def freq_027_feature(text, freq):
	return float(freq['027'])

def freq_suzanne_feature(text, freq):
	return float(freq['suzanne'])

def freq_eb_feature(text, freq):
	return float(freq['eb'])

def freq_kinsey_feature(text, freq):
	return float(freq['kinsey'])

def freq_centana_feature(text, freq):
	return float(freq['centana'])

def freq_5192_feature(text, freq):
	return float(freq['5192'])

def freq_allocations_feature(text, freq):
	return float(freq['allocations'])

def freq_2700_feature(text, freq):
	return float(freq['2700'])

def freq_pena_feature(text, freq):
	return float(freq['pena'])

# Generates a feature vector
def generate_feature_vector(text, freq):
	feature = []
	feature.append(freq_pain_feature(text, freq))
	feature.append(freq_private_feature(text, freq))
	feature.append(freq_bank_feature(text, freq))
	feature.append(freq_money_feature(text, freq))
	feature.append(freq_drug_feature(text, freq))
	feature.append(freq_spam_feature(text, freq))
	feature.append(freq_prescription_feature(text, freq))
	feature.append(freq_creative_feature(text, freq))
	feature.append(freq_height_feature(text, freq))
	feature.append(freq_featured_feature(text, freq))
	feature.append(freq_differ_feature(text, freq))
	feature.append(freq_energy_feature(text, freq))
	feature.append(freq_message_feature(text, freq))
	feature.append(freq_volumes_feature(text, freq))
	feature.append(freq_revision_feature(text, freq))
	feature.append(freq_memo_feature(text, freq))
	feature.append(freq_planning_feature(text, freq))
	feature.append(freq_out_feature(text, freq))

	# Punctuation
	feature.append(freq_semicolon_feature(text, freq))
	feature.append(freq_dollar_feature(text, freq))
	feature.append(freq_sharp_feature(text, freq))
	feature.append(freq_exclamation_feature(text, freq))
	feature.append(freq_para_feature(text, freq))
	feature.append(freq_and_feature(text, freq))
	feature.append(freq_colon_feature(text, freq))
	feature.append(freq_double_quotes_feature(text, freq))
	feature.append(freq_single_quote_feature(text, freq))
	feature.append(freq_front_slash_feature(text, freq))
	feature.append(freq_closing_paren_feature(text, freq))
	feature.append(freq_pipe_feature(text, freq))
	feature.append(freq_dash_feature(text, freq))
	feature.append(freq_caret_feature(text, freq))
	feature.append(freq_at_feature(text, freq))
	feature.append(freq_greater_than_feature(text, freq))

	# Misc
	feature.append(freq_digit_feature(text, freq))

	# --------- Spam features ---------
	feature.append(freq_online_feature(text, freq))
	feature.append(freq_enlarge_feature(text, freq))
	feature.append(freq_click_feature(text, freq))
	feature.append(freq_here_feature(text, freq))
	feature.append(freq_guaranteed_feature(text, freq))
	feature.append(freq_free_feature(text, freq))
	feature.append(freq_please_feature(text, freq))
	# Start here
	feature.append(freq_td_feature(text, freq))
	feature.append(freq_nbsp_feature(text, freq))
	feature.append(freq_pills_feature(text, freq))
	feature.append(freq_width_feature(text, freq))
	feature.append(freq_computron_feature(text, freq))
	feature.append(freq_href_feature(text, freq))
	feature.append(freq_viagra_feature(text, freq))
	feature.append(freq_xp_feature(text, freq))
	feature.append(freq_src_feature(text, freq))
	feature.append(freq_cialis_feature(text, freq))
	feature.append(freq_soft_feature(text, freq))
	feature.append(freq_meds_feature(text, freq))
	feature.append(freq_paliourg_feature(text, freq))
	feature.append(freq_voip_feature(text, freq))
	feature.append(freq_oo_feature(text, freq))
	feature.append(freq_php_feature(text, freq))
	feature.append(freq_bgcolor_feature(text, freq))
	feature.append(freq_drugs_feature(text, freq))
	feature.append(freq_biz_feature(text, freq))
	feature.append(freq_mx_feature(text, freq))
	feature.append(freq_img_feature(text, freq))
	feature.append(freq_photoshop_feature(text, freq))
	feature.append(freq_valign_feature(text, freq))
	feature.append(freq_uncertainties_feature(text, freq))
	feature.append(freq_000000_feature(text, freq))
	feature.append(freq_div_feature(text, freq))
	feature.append(freq_hotlist_feature(text, freq))
	feature.append(freq_moopid_feature(text, freq))
	feature.append(freq_pharmacy_feature(text, freq))
	feature.append(freq_projections_feature(text, freq))
	feature.append(freq_gr_feature(text, freq))
	feature.append(freq_xanax_feature(text, freq))
	feature.append(freq_htmlimg_feature(text, freq))
	feature.append(freq_colspan_feature(text, freq))
	feature.append(freq_corel_feature(text, freq))
	feature.append(freq_readers_feature(text, freq))
	feature.append(freq_macromedia_feature(text, freq))
	feature.append(freq_dealer_feature(text, freq))
	feature.append(freq_knle_feature(text, freq))
	feature.append(freq_valium_feature(text, freq))
	feature.append(freq_demokritos_feature(text, freq))
	feature.append(freq_iit_feature(text, freq))
	feature.append(freq_rolex_feature(text, freq))
	feature.append(freq_intel_feature(text, freq))
	feature.append(freq_speculative_feature(text, freq))
	feature.append(freq_1933_feature(text, freq))
	feature.append(freq_rnd_feature(text, freq))
	feature.append(freq_rx_feature(text, freq))
	feature.append(freq_jebel_feature(text, freq))
	feature.append(freq_lots_feature(text, freq))
	feature.append(freq_half_feature(text, freq))
	feature.append(freq_alt_feature(text, freq))
	feature.append(freq_ooking_feature(text, freq))
	feature.append(freq_darial_feature(text, freq))
	feature.append(freq_yap_feature(text, freq))
	feature.append(freq_gra_feature(text, freq))
	feature.append(freq_0310041_feature(text, freq))
	feature.append(freq_canon_feature(text, freq))
	feature.append(freq_export_feature(text, freq))
	feature.append(freq_1934_feature(text, freq))
	feature.append(freq_0424040_feature(text, freq))
	feature.append(freq_pill_feature(text, freq))
	feature.append(freq_ftar_feature(text, freq))
	feature.append(freq_logos_feature(text, freq))
	feature.append(freq_toshiba_feature(text, freq))
	feature.append(freq_soma_feature(text, freq))
	feature.append(freq_fontfont_feature(text, freq))
	feature.append(freq_materially_feature(text, freq))
	feature.append(freq_vicodin_feature(text, freq))
	feature.append(freq_resellers_feature(text, freq))
	feature.append(freq_viewsonic_feature(text, freq))
	feature.append(freq_cellpadding_feature(text, freq))
	feature.append(freq_1618_feature(text, freq))
	feature.append(freq_robotics_feature(text, freq))
	feature.append(freq_8834464_feature(text, freq))
	feature.append(freq_hewlett_feature(text, freq))
	feature.append(freq_8834454_feature(text, freq))
	feature.append(freq_4176_feature(text, freq))
	feature.append(freq_materia_feature(text, freq))
	feature.append(freq_1226030_feature(text, freq))
	feature.append(freq_apc_feature(text, freq))
	feature.append(freq_iomega_feature(text, freq))
	feature.append(freq_targus_feature(text, freq))
	feature.append(freq_aopen_feature(text, freq))
	feature.append(freq_customerservice_feature(text, freq))
	feature.append(freq_nomad_feature(text, freq))
	feature.append(freq_packard_feature(text, freq))
	feature.append(freq_intellinet_feature(text, freq))
	feature.append(freq_enquiries_feature(text, freq))
	feature.append(freq_construed_feature(text, freq))
	feature.append(freq_drug_feature(text, freq))
	feature.append(freq_uae_feature(text, freq))
	feature.append(freq_itoy_feature(text, freq))
	feature.append(freq_predictions_feature(text, freq))
	feature.append(freq_muscle_feature(text, freq))
	feature.append(freq_zonedubai_feature(text, freq))
	feature.append(freq_emirates_feature(text, freq))
	feature.append(freq_leth_feature(text, freq))
	feature.append(freq_anticipates_feature(text, freq))
	feature.append(freq_aeor_feature(text, freq))
	feature.append(freq_oem_feature(text, freq))
	feature.append(freq_otcbb_feature(text, freq))
	feature.append(freq_assurance_feature(text, freq))
	feature.append(freq_internationa_feature(text, freq))
	feature.append(freq_female_feature(text, freq))
	feature.append(freq_resuits_feature(text, freq))
	feature.append(freq_studio_feature(text, freq))
	feature.append(freq_spain_feature(text, freq))
	feature.append(freq_palestinian_feature(text, freq))
	feature.append(freq_illustrator_feature(text, freq))
	feature.append(freq_verdana_feature(text, freq))
	feature.append(freq_aerofoam_feature(text, freq))
	feature.append(freq_mining_feature(text, freq))
	feature.append(freq_arial_feature(text, freq))
	feature.append(freq_graphics_feature(text, freq))
	feature.append(freq_vi_feature(text, freq))
	feature.append(freq_abdv_feature(text, freq))
	feature.append(freq_wysak_feature(text, freq))
	feature.append(freq_sofftwaares_feature(text, freq))
	feature.append(freq_apple_feature(text, freq))
	feature.append(freq_eogi_feature(text, freq))
	feature.append(freq_erections_feature(text, freq))
	feature.append(freq_three_feature(text, freq))
	feature.append(freq_pubiisher_feature(text, freq))
	feature.append(freq_wiil_feature(text, freq))
	feature.append(freq_gains_feature(text, freq))
	feature.append(freq_ffffff_feature(text, freq))
	feature.append(freq_phentermine_feature(text, freq))
	feature.append(freq_hottlist_feature(text, freq))
	feature.append(freq_cellspacing_feature(text, freq))
	feature.append(freq_male_feature(text, freq))
	feature.append(freq_atleast_feature(text, freq))
	feature.append(freq_dosage_feature(text, freq))
	feature.append(freq_netherlands_feature(text, freq))
	feature.append(freq_ail_feature(text, freq))
	feature.append(freq_prescriptions_feature(text, freq))
	feature.append(freq_undervalued_feature(text, freq))
	feature.append(freq_cia_feature(text, freq))
	feature.append(freq_couid_feature(text, freq))
	feature.append(freq_technoiogies_feature(text, freq))
	feature.append(freq_dose_feature(text, freq))
	feature.append(freq_opinions_feature(text, freq))
	feature.append(freq_sir_feature(text, freq))
	feature.append(freq_ur_feature(text, freq))
	feature.append(freq_kin_feature(text, freq))
	feature.append(freq_discreet_feature(text, freq))
	feature.append(freq_prozac_feature(text, freq))
	feature.append(freq_brbr_feature(text, freq))
	feature.append(freq_tirr_feature(text, freq))
	feature.append(freq_serial_feature(text, freq))
	feature.append(freq_tadalafil_feature(text, freq))
	feature.append(freq_pinnacle_feature(text, freq))
	feature.append(freq_distributorjebel_feature(text, freq))
	feature.append(freq_mai_feature(text, freq))
	feature.append(freq_artprice_feature(text, freq))
	feature.append(freq_iso_feature(text, freq))
	feature.append(freq_ambien_feature(text, freq))
	feature.append(freq_cheapest_feature(text, freq))
	feature.append(freq_vlagra_feature(text, freq))
	feature.append(freq_lasts_feature(text, freq))
	feature.append(freq_microcap_feature(text, freq))
	feature.append(freq_techlite_feature(text, freq))
	feature.append(freq_accessories_feature(text, freq))
	feature.append(freq_indicative_feature(text, freq))
	feature.append(freq_impotence_feature(text, freq))
	feature.append(freq_levitra_feature(text, freq))
	feature.append(freq_alcohol_feature(text, freq))
	feature.append(freq_lls_feature(text, freq))
	feature.append(freq_manufacturer_feature(text, freq))
	feature.append(freq_nigeria_feature(text, freq))
	feature.append(freq_imited_feature(text, freq))
	feature.append(freq_pr_feature(text, freq))
	feature.append(freq_emerson_feature(text, freq))
	feature.append(freq_erection_feature(text, freq))
	feature.append(freq_der_feature(text, freq))
	feature.append(freq_es_feature(text, freq))
	feature.append(freq_israeli_feature(text, freq))
	feature.append(freq_constitutes_feature(text, freq))
	feature.append(freq_ooo_feature(text, freq))
	feature.append(freq_spacer_feature(text, freq))
	feature.append(freq_piease_feature(text, freq))
	feature.append(freq_porn_feature(text, freq))
	feature.append(freq_bingo_feature(text, freq))
	feature.append(freq_paypal_feature(text, freq))
	feature.append(freq_reform_feature(text, freq))
	feature.append(freq_verge_feature(text, freq))
	feature.append(freq_vl_feature(text, freq))
	feature.append(freq_fontbr_feature(text, freq))
	feature.append(freq_ffffffstrongfont_feature(text, freq))
	feature.append(freq_sans_feature(text, freq))
	feature.append(freq_macintosh_feature(text, freq))
	feature.append(freq_newsietter_feature(text, freq))
	feature.append(freq_bias_feature(text, freq))
	feature.append(freq_omit_feature(text, freq))
	feature.append(freq_packaging_feature(text, freq))
	feature.append(freq_replica_feature(text, freq))
	feature.append(freq_genuine_feature(text, freq))
	feature.append(freq_cum_feature(text, freq))
	feature.append(freq_ion_feature(text, freq))
	feature.append(freq_biosphere_feature(text, freq))
	# --------- Ham features ---------
	feature.append(freq_enron_feature(text, freq))
	feature.append(freq_meter_feature(text, freq))
	feature.append(freq_hpl_feature(text, freq))
	feature.append(freq_daren_feature(text, freq))
	feature.append(freq_mmbtu_feature(text, freq))
	feature.append(freq_xls_feature(text, freq))
	feature.append(freq_sitara_feature(text, freq))
	feature.append(freq_volumes_feature(text, freq))
	feature.append(freq_pec_feature(text, freq))
	feature.append(freq_ena_feature(text, freq))
	feature.append(freq_melissa_feature(text, freq))
	feature.append(freq_teco_feature(text, freq))
	feature.append(freq_tenaska_feature(text, freq))
	feature.append(freq_pat_feature(text, freq))
	feature.append(freq_aimee_feature(text, freq))
	feature.append(freq_noms_feature(text, freq))
	feature.append(freq_actuals_feature(text, freq))
	feature.append(freq_hsc_feature(text, freq))
	feature.append(freq_cotten_feature(text, freq))
	feature.append(freq_chokshi_feature(text, freq))
	feature.append(freq_fyi_feature(text, freq))
	feature.append(freq_hplc_feature(text, freq))
	feature.append(freq_wellhead_feature(text, freq))
	feature.append(freq_clynes_feature(text, freq))
	feature.append(freq_eastrans_feature(text, freq))
	feature.append(freq_counterparty_feature(text, freq))
	feature.append(freq_txu_feature(text, freq))
	feature.append(freq_hplno_feature(text, freq))
	feature.append(freq_rita_feature(text, freq))
	feature.append(freq_lannou_feature(text, freq))
	feature.append(freq_nominations_feature(text, freq))
	feature.append(freq_pefs_feature(text, freq))
	feature.append(freq_enronxgate_feature(text, freq))
	feature.append(freq_weissman_feature(text, freq))
	feature.append(freq_gcs_feature(text, freq))
	feature.append(freq_cec_feature(text, freq))
	feature.append(freq_wynne_feature(text, freq))
	feature.append(freq_hplo_feature(text, freq))
	feature.append(freq_allocated_feature(text, freq))
	feature.append(freq_iferc_feature(text, freq))
	feature.append(freq_path_feature(text, freq))
	feature.append(freq_spreadsheet_feature(text, freq))
	feature.append(freq_anita_feature(text, freq))
	feature.append(freq_buyback_feature(text, freq))
	feature.append(freq_equistar_feature(text, freq))
	feature.append(freq_sherlyn_feature(text, freq))
	feature.append(freq_flowed_feature(text, freq))
	feature.append(freq_pops_feature(text, freq))
	feature.append(freq_scheduling_feature(text, freq))
	feature.append(freq_entex_feature(text, freq))
	feature.append(freq_katy_feature(text, freq))
	feature.append(freq_ees_feature(text, freq))
	feature.append(freq_clem_feature(text, freq))
	feature.append(freq_darren_feature(text, freq))
	feature.append(freq_calpine_feature(text, freq))
	feature.append(freq_gco_feature(text, freq))
	feature.append(freq_aep_feature(text, freq))
	feature.append(freq_midcon_feature(text, freq))
	feature.append(freq_cornhusker_feature(text, freq))
	feature.append(freq_redeliveries_feature(text, freq))
	feature.append(freq_victor_feature(text, freq))
	feature.append(freq_schumack_feature(text, freq))
	feature.append(freq_reinhardt_feature(text, freq))
	feature.append(freq_luong_feature(text, freq))
	feature.append(freq_lsk_feature(text, freq))
	feature.append(freq_herod_feature(text, freq))
	feature.append(freq_hplnl_feature(text, freq))
	feature.append(freq_methanol_feature(text, freq))
	feature.append(freq_revision_feature(text, freq))
	feature.append(freq_6353_feature(text, freq))
	feature.append(freq_papayoti_feature(text, freq))
	feature.append(freq_dfarmer_feature(text, freq))
	feature.append(freq_bryan_feature(text, freq))
	feature.append(freq_lamphier_feature(text, freq))
	feature.append(freq_valero_feature(text, freq))
	feature.append(freq_cleburne_feature(text, freq))
	feature.append(freq_baumbach_feature(text, freq))
	feature.append(freq_poorman_feature(text, freq))
	feature.append(freq_liz_feature(text, freq))
	feature.append(freq_eol_feature(text, freq))
	feature.append(freq_lsp_feature(text, freq))
	feature.append(freq_waha_feature(text, freq))
	feature.append(freq_riley_feature(text, freq))
	feature.append(freq_enrononline_feature(text, freq))
	feature.append(freq_enserch_feature(text, freq))
	feature.append(freq_outage_feature(text, freq))
	feature.append(freq_employee_feature(text, freq))
	feature.append(freq_098_feature(text, freq))
	feature.append(freq_cernosek_feature(text, freq))
	feature.append(freq_pathed_feature(text, freq))
	feature.append(freq_853_feature(text, freq))
	feature.append(freq_aepin_feature(text, freq))
	feature.append(freq_tejas_feature(text, freq))
	feature.append(freq_saturday_feature(text, freq))
	feature.append(freq_withers_feature(text, freq))
	feature.append(freq_boas_feature(text, freq))
	feature.append(freq_superty_feature(text, freq))
	feature.append(freq_avila_feature(text, freq))
	feature.append(freq_easttexas_feature(text, freq))
	feature.append(freq_hesco_feature(text, freq))
	feature.append(freq_lamadrid_feature(text, freq))
	feature.append(freq_cdnow_feature(text, freq))
	feature.append(freq_herrera_feature(text, freq))
	feature.append(freq_gpgfin_feature(text, freq))
	feature.append(freq_hakemack_feature(text, freq))
	feature.append(freq_sandi_feature(text, freq))
	feature.append(freq_paso_feature(text, freq))
	feature.append(freq_cp_feature(text, freq))
	feature.append(freq_kcs_feature(text, freq))
	feature.append(freq_eileen_feature(text, freq))
	feature.append(freq_reliantenergy_feature(text, freq))
	feature.append(freq_4179_feature(text, freq))
	feature.append(freq_comments_feature(text, freq))
	feature.append(freq_9497_feature(text, freq))
	feature.append(freq_345_feature(text, freq))
	feature.append(freq_gomes_feature(text, freq))
	feature.append(freq_ponton_feature(text, freq))
	feature.append(freq_enw_feature(text, freq))
	feature.append(freq_1266_feature(text, freq))
	feature.append(freq_enerfin_feature(text, freq))
	feature.append(freq_neon_feature(text, freq))
	feature.append(freq_tisdale_feature(text, freq))
	feature.append(freq_dynegy_feature(text, freq))
	feature.append(freq_strangers_feature(text, freq))
	feature.append(freq_tetco_feature(text, freq))
	feature.append(freq_mckay_feature(text, freq))
	feature.append(freq_invoices_feature(text, freq))
	feature.append(freq_neuweiler_feature(text, freq))
	feature.append(freq_intrastate_feature(text, freq))
	feature.append(freq_mops_feature(text, freq))
	feature.append(freq_charlene_feature(text, freq))
	feature.append(freq_trader_feature(text, freq))
	feature.append(freq_interconnect_feature(text, freq))
	feature.append(freq_0435_feature(text, freq))
	feature.append(freq_gtc_feature(text, freq))
	feature.append(freq_olsen_feature(text, freq))
	feature.append(freq_availabilities_feature(text, freq))
	feature.append(freq_tufco_feature(text, freq))
	feature.append(freq_panenergy_feature(text, freq))
	feature.append(freq_heidi_feature(text, freq))
	feature.append(freq_valadez_feature(text, freq))
	feature.append(freq_billed_feature(text, freq))
	feature.append(freq_77002_feature(text, freq))
	feature.append(freq_cass_feature(text, freq))
	feature.append(freq_revisions_feature(text, freq))
	feature.append(freq_unaccounted_feature(text, freq))
	feature.append(freq_brazos_feature(text, freq))
	feature.append(freq_copano_feature(text, freq))
	feature.append(freq_origination_feature(text, freq))
	feature.append(freq_winfree_feature(text, freq))
	feature.append(freq_epgt_feature(text, freq))
	feature.append(freq_christy_feature(text, freq))
	feature.append(freq_reveffo_feature(text, freq))
	feature.append(freq_crosstex_feature(text, freq))
	feature.append(freq_troy_feature(text, freq))
	feature.append(freq_dth_feature(text, freq))
	feature.append(freq_lonestar_feature(text, freq))
	feature.append(freq_rick_feature(text, freq))
	feature.append(freq_encina_feature(text, freq))
	feature.append(freq_kelly_feature(text, freq))
	feature.append(freq_ews_feature(text, freq))
	feature.append(freq_mmbtus_feature(text, freq))
	feature.append(freq_cpr_feature(text, freq))
	feature.append(freq_mtr_feature(text, freq))
	feature.append(freq_csikos_feature(text, freq))
	feature.append(freq_goliad_feature(text, freq))
	feature.append(freq_pager_feature(text, freq))
	feature.append(freq_assigned_feature(text, freq))
	feature.append(freq_vols_feature(text, freq))
	feature.append(freq_mcmills_feature(text, freq))
	feature.append(freq_beaty_feature(text, freq))
	feature.append(freq_marta_feature(text, freq))
	feature.append(freq_kristen_feature(text, freq))
	feature.append(freq_gdp_feature(text, freq))
	feature.append(freq_gisb_feature(text, freq))
	feature.append(freq_henderson_feature(text, freq))
	feature.append(freq_463_feature(text, freq))
	feature.append(freq_sweeney_feature(text, freq))
	feature.append(freq_texoma_feature(text, freq))
	feature.append(freq_hernandez_feature(text, freq))
	feature.append(freq_ebs_feature(text, freq))
	feature.append(freq_hillary_feature(text, freq))
	feature.append(freq_seaman_feature(text, freq))
	feature.append(freq_schedulers_feature(text, freq))
	feature.append(freq_patti_feature(text, freq))
	feature.append(freq_lindley_feature(text, freq))
	feature.append(freq_memo_feature(text, freq))
	feature.append(freq_mazowita_feature(text, freq))
	feature.append(freq_pathing_feature(text, freq))
	feature.append(freq_984132_feature(text, freq))
	feature.append(freq_pgev_feature(text, freq))
	feature.append(freq_027_feature(text, freq))
	feature.append(freq_suzanne_feature(text, freq))
	feature.append(freq_eb_feature(text, freq))
	feature.append(freq_kinsey_feature(text, freq))
	feature.append(freq_centana_feature(text, freq))
	feature.append(freq_5192_feature(text, freq))
	feature.append(freq_allocations_feature(text, freq))
	feature.append(freq_2700_feature(text, freq))
	feature.append(freq_pena_feature(text, freq))

	return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
	design_matrix = []
	for filename in filenames:
		with open(filename) as f:
			text = f.read() # Read in text from file
			text = text.replace('\r\n', ' ') # Remove newline character
			words = re.findall(r'\w+', text)
			word_freq = defaultdict(int) # Frequency of all words
			for word in words:
				word_freq[word] += 1

			# Create a feature vector
			feature_vector = generate_feature_vector(text, word_freq)
			design_matrix.append(feature_vector)
	return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = [1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)

