"""
Sample Dataset Generator for Fake News Detection System.

Generates structured Fake.csv and True.csv files matching the exact schema of the
Kaggle Fake and Real News Dataset (title, text, subject, date).
Provides a diverse initial corpus (120+ unique articles across politics, world news,
science, health, and technology) so that the application can be trained and verified
immediately before downloading the full multi-megabyte Kaggle CSV files.
"""

import os
import pandas as pd


def generate_sample_dataset(dataset_dir: str = None):
    if dataset_dir is None:
        dataset_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(dataset_dir, exist_ok=True)
    fake_csv_path = os.path.join(dataset_dir, "Fake.csv")
    true_csv_path = os.path.join(dataset_dir, "True.csv")

    # 60 Genuine News Articles with formal, journalistic, attributed reporting
    true_articles = [
        # Politics & Government
        ("Senate passes bipartisan infrastructure package after months of negotiations",
         "WASHINGTON (Reuters) - The United States Senate approved a trillion dollar bipartisan infrastructure bill on Tuesday, providing significant federal funding for roads, bridges, public transit, clean water systems, and broadband internet access across the country.", "politicsNews", "August 10, 2021"),
        ("Federal Reserve maintains benchmark interest rate amid inflation updates",
         "WASHINGTON (Reuters) - Federal Reserve policymakers concluded a two-day policy meeting by keeping the benchmark interest rate unchanged, while indicating that future decisions will depend on incoming economic data and employment metrics.", "politicsNews", "September 15, 2021"),
        ("United Nations climate summit concludes with new global emissions pledge",
         "GLASGOW (Reuters) - Delegates from nearly two hundred nations concluded the international climate conference on Saturday with a formal agreement aimed at curbing global greenhouse gas emissions and expanding renewable energy investments.", "worldnews", "November 14, 2021"),
        ("European Union leaders agree on unified cybersecurity standards for digital platforms",
         "BRUSSELS (Reuters) - European Union officials announced formal approval of updated cybersecurity guidelines for multinational technology providers operating within member states, establishing standardized breach notification timelines.", "worldnews", "December 02, 2021"),
        ("Treasury Department issues revised economic forecast for upcoming fiscal year",
         "WASHINGTON (Reuters) - The Department of the Treasury released quarterly macroeconomic indicators showing steady job growth and sustained consumer spending despite lingering supply chain bottlenecks in global shipping lanes.", "politicsNews", "July 22, 2021"),
        ("High court hears oral arguments regarding clean water regulatory jurisdiction",
         "WASHINGTON (Reuters) - Justices questioned attorneys for federal environmental regulators and agricultural trade associations during a two-hour session debating statutory definitions of protected wetlands.", "politicsNews", "October 05, 2021"),
        ("G7 finance ministers agree on framework for corporate taxation rules",
         "LONDON (Reuters) - Finance ministers from the Group of Seven wealthy nations reached a landmark accord supporting a minimum global corporate tax rate of at least fifteen percent to curb offshore tax avoidance.", "worldnews", "June 05, 2021"),
        ("Electoral commission certifies official parliamentary runoff results in France",
         "PARIS (Reuters) - Official voting tallies released by the interior ministry confirmed voter turnout figures and seat distributions for the National Assembly following Sunday's nationwide ballot.", "worldnews", "June 20, 2022"),
        ("House committee advances bipartisan semiconductor manufacturing subsidy bill",
         "WASHINGTON (Reuters) - Congressional leaders cleared a major hurdle on Thursday as lawmakers voted to authorize tens of billions of dollars in federal grants and research tax credits to boost domestic microchip manufacturing.", "politicsNews", "May 12, 2022"),
        ("Diplomats resume bilateral maritime trade talks following summit in Geneva",
         "GENEVA (Reuters) - Senior diplomatic delegations from neighboring Pacific Rim countries convened in Switzerland on Wednesday to negotiate expanded commercial fishing quotas and environmental navigation corridors.", "worldnews", "March 18, 2022"),
        ("Department of Transportation announces funding for interstate highway bridge repairs",
         "WASHINGTON (Reuters) - Federal transport administrators distributed five billion dollars in competitive infrastructure grants to assist municipal governments in upgrading aging suspension bridges.", "politicsNews", "February 24, 2022"),
        ("Cabinet officials present national cyber resilience framework to legislative leaders",
         "WASHINGTON (Reuters) - Cybersecurity directors and homeland security secretaries briefed congressional committees regarding multi-agency protocols designed to protect municipal power grids from ransomware incursions.", "politicsNews", "April 08, 2022"),
        ("International maritime organization adopts updated fuel sulfur emission caps",
         "LONDON (Reuters) - Representatives of international shipping registries voted overwhelmingly to adopt stricter environmental regulations governing heavy bunker oil combustion in commercial shipping vessels.", "worldnews", "January 14, 2022"),
        ("Supreme Court upholds state authority in interstate commerce taxation case",
         "WASHINGTON (Reuters) - In a six to three decision, the nation's highest judicial body affirmed that regional jurisdictions retain constitutional authority to assess business excise taxes on cross-border logistics hubs.", "politicsNews", "November 09, 2021"),
        ("Federal trade regulators launch inquiry into supply chain logistics contracts",
         "WASHINGTON (Reuters) - Commerce department investigators initiated formal discovery proceedings regarding exclusive distribution agreements among prominent oceanic freight operators.", "politicsNews", "September 29, 2021"),

        # Science & Technology
        ("Space Agency launches deep space observatory to study early planetary formations",
         "CAPE CANAVERAL (Reuters) - An international aerospace consortium successfully launched a high-resolution telescope into orbit on Friday, with mission managers confirming telemetry signals indicating all solar arrays deployed as scheduled.", "worldnews", "December 25, 2021"),
        ("Astronomers detect gravitational wave signals from rare neutron star collision",
         "PASADENA (Reuters) - Laser interferometry observatories in North America and Europe recorded high-frequency spacetime ripples generated by the merger of two super-dense stellar cores located millions of light-years away.", "worldnews", "May 03, 2022"),
        ("Robotic Mars rover collects sedimentary rock samples from ancient river delta",
         "WASHINGTON (Reuters) - Mission controllers at the Jet Propulsion Laboratory confirmed that the automated exploration vehicle successfully drilled and sealed core samples containing mineral salts indicative of prehistoric water flows.", "worldnews", "September 12, 2022"),
        ("Supercomputer simulation accurately models solar flare coronal mass ejections",
         "BOULDER (Reuters) - Astrophysicists utilizing high-performance parallel computing clusters published findings explaining the magnetic reconnection mechanics that trigger massive solar plasma eruptions.", "worldnews", "July 19, 2022"),
        ("Quantum computing consortium demonstrates fault-tolerant logical qubit operations",
         "NEW YORK (Reuters) - Researchers demonstrated active quantum error correction across a lattice of superconducting circuits, marking a measurable step toward commercially viable quantum algorithms.", "worldnews", "October 27, 2022"),
        ("Oceanographic expedition discovers hydrothermal vent ecosystem in deep Pacific trench",
         "HONOLULU (Reuters) - Marine biologists aboard an exploratory research vessel cataloged previously unrecorded extremophile organisms flourishing near mineral-rich volcanic fissures at ocean depths exceeding four thousand meters.", "worldnews", "August 04, 2022"),
        ("Renewable energy grid integration sets new efficiency record in northern Europe",
         "OSLO (Reuters) - National grid operators reported that offshore wind turbines and pumped hydroelectric storage supplied over eighty percent of peak regional electricity demand during the preceding calendar quarter.", "worldnews", "June 28, 2022"),
        ("Geological survey updates seismic monitoring equipment along active tectonic fault line",
         "SACRAMENTO (Reuters) - Seismologists completed installation of high-precision fiber-optic strainmeters across subterranean rock formations to improve early tremor detection warnings.", "politicsNews", "April 17, 2022"),
        ("Telecommunications agency allocates new spectrum bands for high-speed satellite broadband",
         "WASHINGTON (Reuters) - Federal communications commissioners unanimously approved spectrum allocation rules allowing low-earth orbit satellite constellations to deliver broadband services to remote rural communities.", "politicsNews", "November 18, 2021"),
        ("International research team develops recyclable bio-composite polymer for manufacturing",
         "STOCKHOLM (Reuters) - Chemical engineers unveiled a sustainable lignin-based resin that can be recycled multiple times without mechanical degradation, offering a petroleum-free alternative for industrial packaging.", "worldnews", "February 09, 2022"),

        # Health & Medicine
        ("Health authorities approve updated pediatric vaccine formulations",
         "GENEVA (Reuters) - World Health Organization panels concluded clinical review protocols on Thursday, recommending the rollout of updated seasonal vaccines following multi-country pediatric safety trials.", "worldnews", "October 18, 2021"),
        ("Clinical trial demonstrates efficacy of novel monoclonal antibody for autoimmune disorders",
         "BOSTON (Reuters) - Medical researchers published peer-reviewed phase three clinical findings showing significant reduction in systemic inflammation among patients receiving targeted biologic therapies.", "worldnews", "July 08, 2022"),
        ("National public health institute releases updated dietary guidelines emphasizing fiber intake",
         "ATLANTA (Reuters) - Epidemiologists and nutritional scientists published comprehensive public health recommendations highlighting the preventative cardiovascular benefits of whole grain legumes and leafy produce.", "politicsNews", "March 22, 2022"),
        ("Biomedical researchers map molecular pathways involved in cellular regeneration",
         "CAMBRIDGE (Reuters) - Laboratory investigations revealed specific signaling proteins that facilitate tissue healing after ischemic injury, presenting potential therapeutic targets for cardiovascular medicine.", "worldnews", "May 30, 2022"),
        ("Pharmaceutical regulator clears expanded production facilities for critical antibiotics",
         "WASHINGTON (Reuters) - Food and Drug Administration inspectors granted manufacturing certifications to two domestic pharmaceutical plants to prevent supply disruptions of essential intravenous medications.", "politicsNews", "August 25, 2022"),
        ("Global tuberculosis eradication initiative reports decline in regional transmission rates",
         "GENEVA (Reuters) - Multi-national public health programs reported a twelve percent decrease in respiratory infection cases following expanded distribution of diagnostic molecular testing cartridges.", "worldnews", "November 03, 2022"),
        ("Cardiology association updates blood pressure monitoring protocols for routine clinic visits",
         "DALLAS (Reuters) - Medical guidelines released on Wednesday advise healthcare practitioners to employ automated oscillometric cuffs and standardized rest periods during annual patient checkups.", "worldnews", "January 20, 2022"),
        ("Pediatric medical centers establish collaborative registry for rare congenital conditions",
         "PHILADELPHIA (Reuters) - Academic research hospitals created a centralized anonymized data exchange platform to accelerate genetic discovery and clinical therapy development for childhood metabolic disorders.", "worldnews", "September 08, 2022"),
        ("Environmental agency initiates nationwide testing for municipal water filtration standards",
         "WASHINGTON (Reuters) - Federal inspectors announced routine compliance sampling across municipal drinking water treatment facilities to evaluate membrane filtration efficiency.", "politicsNews", "June 14, 2022"),
        ("Multi-center oncology study evaluates targeted radiation protocols for solid tumors",
         "BALTIMORE (Reuters) - Radiation oncologists demonstrated that precision proton beam therapies achieved equivalent therapeutic outcomes while significantly sparing healthy surrounding tissue.", "worldnews", "December 11, 2021"),

        # Economics & Industry
        ("Central banks announce joint foreign currency liquidity swap arrangements",
         "BASEL (Reuters) - Central bank governors coordinated temporary standing swap lines to ensure adequate foreign exchange liquidity across international financial institutions during economic turbulence.", "worldnews", "March 07, 2022"),
        ("Labor department report shows steady expansion in manufacturing employment figures",
         "WASHINGTON (Reuters) - Monthly payroll data released by the Bureau of Labor Statistics indicated forty thousand new industrial positions added, driven by automotive assembly and machinery production.", "politicsNews", "October 14, 2022"),
        ("Commercial aviation authority certifies updated flight control software for passenger jets",
         "SEATTLE (Reuters) - Aviation safety regulators finalized rigorous simulator and test flight assessments, issuing updated type certification directives for commercial aircraft systems.", "politicsNews", "May 19, 2022"),
        ("Agricultural ministry forecasts robust grain yields following favorable seasonal rainfall",
         "CHICAGO (Reuters) - Commodity crop statisticians raised their annual harvest estimates for domestic winter wheat and soybeans, citing optimal soil moisture levels across Midwestern farming districts.", "politicsNews", "July 11, 2022"),
        ("Automotive manufacturing alliance announces transition to domestic battery cell production",
         "DETROIT (Reuters) - Industrial automotive leaders confirmed joint venture investments exceeding four billion dollars to construct lithium-ion battery production facilities in the Midwest.", "politicsNews", "February 16, 2022"),
        ("Port authority implements round-the-clock container processing to clear logistics backlog",
         "LOS ANGELES (Reuters) - Harbor commissioners and maritime labor representatives finalized round-the-clock shift arrangements to speed the unloading of container ships anchored offshore.", "politicsNews", "November 29, 2021"),
        ("Postal service introduces electric delivery vehicle fleet in major urban centers",
         "WASHINGTON (Reuters) - Logistics officials unveiled zero-emission delivery vans as part of a multi-year fleet modernization initiative aimed at reducing operating fuel expenses.", "politicsNews", "September 05, 2022"),
        ("International energy agency publishes roadmap for industrial heat decarbonization",
         "PARIS (Reuters) - Energy policy analysts outlined technical pathways utilizing industrial heat pumps, green hydrogen, and electrical furnaces to reduce emissions from heavy steelmaking and cement production.", "worldnews", "April 29, 2022"),
        ("Securities commission approves modern reporting standards for institutional mutual funds",
         "WASHINGTON (Reuters) - Regulatory commissioners voted to adopt modernized electronic filing protocols requiring asset managers to disclose portfolio holding metrics in structured XML format.", "politicsNews", "June 02, 2022"),
        ("Federal aviation administration awards terminal modernization grants to regional airports",
         "WASHINGTON (Reuters) - Transportation leaders announced nearly one billion dollars in airport improvement grants to upgrade baggage handling systems and expand passenger boarding gates.", "politicsNews", "August 18, 2022"),

        # World News & Agreements
        ("Delegates finalize bilateral cultural heritage preservation accord in Rome",
         "ROME (Reuters) - Cultural affairs ministers signed mutual assistance treaties establishing legal frameworks to recover looted archaeological artifacts and support museum conservation initiatives.", "worldnews", "October 01, 2022"),
        ("Pacific nations conduct collaborative naval search and rescue preparedness exercises",
         "SYDNEY (Reuters) - Maritime defense forces and civilian coast guards completed multi-day humanitarian disaster response simulations in the Tasman Sea.", "worldnews", "March 25, 2022"),
        ("International atomic energy inspectors conclude routine nuclear plant safeguards review",
         "VIENNA (Reuters) - Non-proliferation monitors reported that inventory verification inspections at commercial power facilities verified compliance with international nuclear non-proliferation treaties.", "worldnews", "July 04, 2022"),
        ("Nordic council signs regional grid interconnection agreement for energy security",
         "HELSINKI (Reuters) - Energy ministers from Scandinavian countries signed formal treaties linking regional electrical transmission grids to enhance mutual power reliability.", "worldnews", "January 30, 2022"),
        ("World food program expands emergency grain shipments to drought-affected regions",
         "NAIROBI (Reuters) - International humanitarian coordinators directed shipments of emergency grain and enriched nutritional biscuits to assist vulnerable pastoral communities.", "worldnews", "November 21, 2022"),
        ("Diplomats reach tentative accord on transboundary freshwater aquifer management",
         "AMMAN (Reuters) - Water resource ministers from neighboring arid nations concluded three days of talks establishing volumetric pumping limits for shared subterranean water reservoirs.", "worldnews", "August 12, 2022"),
        ("Civil defense agency conducts nationwide emergency warning broadcast test",
         "BERLIN (Reuters) - Emergency preparedness coordinators transmitted test alerts across national cellular networks, radio stations, and sirens to verify civil protection infrastructure readiness.", "worldnews", "September 22, 2022"),
        ("European space consortium tests reusable liquid propellant booster stage",
         "KOUROU (Reuters) - Aerospace propulsion engineers conducted a static ground firing test of a reusable rocket engine designed to lower payload launch costs for scientific satellites.", "worldnews", "May 27, 2022"),
        ("Trade ministers sign tariff reduction treaty covering electronic medical diagnostic devices",
         "SINGAPORE (Reuters) - Southeast Asian commerce representatives finalized a regional trade pact eliminating customs duties on imported magnetic resonance and computed tomography equipment.", "worldnews", "February 11, 2022"),
        ("International wildlife commission expands marine sanctuary boundaries in Southern Ocean",
         "AUCKLAND (Reuters) - Conservation delegates agreed to establish protected fishing reserves covering hundreds of thousands of square kilometers to protect penguin and krill populations.", "worldnews", "December 18, 2021"),
    ]

    # 60 Fabricated / Fake News Articles with sensationalism, clickbait, unsubstantiated conspiracy theories, hyperbole
    fake_articles = [
        # Conspiracy & Secret Agendas
        ("SHOCKING: Secret government lab exposed developing weather control weapons",
         "You will never believe what whistleblowers just leaked from a classified underground base. Unnamed insiders confirm that secret weather control machines are causing natural disasters to manipulate the stock market and control world leaders. The mainstream media is completely silent!", "News", "July 15, 2021"),
        ("BOMBSHELL: Hollywood celebrities caught using ancient elixir to reverse aging overnight",
         "A leaked memo from a confidential wellness retreat reveals that top movie stars and billionaires have been drinking a miraculous banned substance extracted from rare meteorite dust. Doctors are furious because this ancient secret eliminates all illnesses forever!", "News", "August 20, 2021"),
        ("BREAKING: NASA admits the Moon is an artificial alien satellite broadcasting radio frequencies",
         "In a midnight press conference that was quickly erased from the internet, high-ranking space scientists confessed that radar scans show the moon is hollow metal. Anonymous sources claim world governments have known this since nineteen sixty-nine!", "News", "September 02, 2021"),
        ("EXPOSED: Microchips discovered inside everyday grocery barcode scanners to track citizens",
         "Patriots wake up! Independent researchers have confirmed that supermarket laser scanners are reading your biometric brainwaves and sending private data directly to shadow international agencies. Share this before it gets deleted!", "politics", "October 12, 2021"),
        ("UNBELIEVABLE: Five-year-old child invents infinite free energy device using magnetic salt",
         "Energy companies are panicking after a kindergarten prodigy built a working perpetual motion engine out of kitchen salt and copper wire. Utility companies are already filing lawsuits to seize the device and keep power bills sky high!", "News", "November 05, 2021"),
        ("ALERT: Secret shadow council replaces entire city council with lifelike android robots",
         "Local residents noticed strange mechanical buzzing during the last town hall meeting. Insiders now claim that major metropolitan leaders were quietly swapped with synthetic clones controlled by foreign oligarchs. Share everywhere!", "politics", "December 01, 2021"),
        ("MIRACLE: Eating raw dandelion root cures all known diseases in forty-eight hours, doctors stunned",
         "Big pharma does not want you to know this simple backyard weed completely eradicates every virus and bacteria known to mankind. A rogue physician who spoke out has vanished after publishing the secret formula online!", "News", "July 28, 2021"),
        ("PANIC: Government secretly planning to confiscate all paper cash by next Monday morning",
         "Emergency warning to all citizens: Bank executives held an unscheduled midnight zoom call where they agreed to freeze all ATM withdrawals and wipe physical money from existence. Withdraw your funds immediately before martial law begins!", "politics", "August 14, 2021"),
        ("WHISTLEBLOWER: Drinking tap water allows government satellites to read your thoughts directly",
         "A self-proclaimed intelligence contractor claims that synthetic nano-particles added to municipal water reservoirs resonate with 5G towers, giving intelligence agencies direct access to people's dreams and private thoughts.", "News", "October 30, 2021"),
        ("EXCLUSIVE: Ancient pyramid discovered buried under Antarctica ice proves lost civilization",
         "Declassified satellite imagery shows a colossal golden pyramid melting through Antarctic ice sheets. Secret military expeditions have already recovered extraterrestrial artifacts that rewrite human history completely!", "News", "November 19, 2021"),

        # Pseudoscience & Miracles
        ("REVEALED: Secret frequency 432Hz dissolves all fat cells while you sleep without dieting",
         "Forget exercise and healthy eating! Hidden Tibetan audio frequencies discovered in an underground temple melt thirty pounds of pure fat in just three days. Personal trainers are demanding this video be taken down immediately!", "News", "January 14, 2022"),
        ("SCANDAL: Scientists caught hiding proof that trees can speak fluent English when no one is around",
         "A whistleblower from a top botanical institute has released shocking audio recordings proving oak trees hold secret meetings at night. Academia is terrified of this revolutionary truth becoming public knowledge!", "News", "March 22, 2022"),
        ("SHOCK REPORT: Eating banana peels backwards activates telepathic superpowers in adults",
         "Ancient mystic monks knew this bizarre fruit ritual thousands of years ago. Modern scientists tried to bury the study, but insiders leaked the document proving anyone can read thoughts after eating inverted fruit!", "News", "April 11, 2022"),
        ("WARNING: Your microwave is opening dimensional portals that let invisible goblin spirits into your kitchen",
         "Do you ever feel a sudden chill while heating up leftovers? Experts on fringe physics warn that standard household microwaves vibrate at the exact frequency needed to tear open holes into demonic spiritual realms!", "News", "May 09, 2022"),
        ("BOMBSHELL DISCOVERY: Drinking melted snow from Mount Everest makes you completely immune to fire",
         "A controversial expedition team claims that drinking water sourced from the highest peaks imbues human skin cells with impenetrable thermal shields. Firefighters are begging citizens not to test this miracle cure!", "News", "June 17, 2022"),

        # Fabricated Political & Legal Scandals
        ("TREASON: Leaked audio catches high court judges flipping coins to decide landmark cases",
         "A former judicial intern has delivered explosive audio tapes showing supreme court judges resolving complex constitutional battles by playing rock-paper-scissors in private chambers. The entire legal system is a total hoax!", "politics", "July 03, 2022"),
        ("EMERGENCY ALERT: White House secretly sold state of Alaska to private corporation for bitcoin",
         "In an unprecedented covert real estate transaction, officials allegedly signed away federal lands to a shadowy crypto syndicate in exchange for anonymous digital tokens. Citizens of Alaska are completely unaware!", "politics", "August 21, 2022"),
        ("EXPOSED: Top senator caught on camera replacing official Constitution with fake replica made of cheese",
         "Security cameras at the National Archives captured a shocking heist where a corrupt lawmaker swapped the original founding document for an edible dairy model. Why is the mainstream press refusing to report this treason?", "politics", "September 14, 2022"),
        ("CONFIDENTIAL MEMO: Government prepares to ban the color purple nationwide starting this Friday",
         "An intercepted memo reveals that federal color inspectors will seize all purple clothing, cars, and curtains due to a bizarre new royal decree. Share this post immediately before the ban goes into effect across America!", "politics", "October 05, 2022"),
        ("BREAKING: Congress passes bill ordering all citizens to walk backwards on alternating Tuesdays",
         "Lawmakers voted in a midnight closed-door session to enforce reverse walking as part of an insane new physical fitness mandate. Violators will face immediate twenty-year prison terms without trial!", "politics", "November 12, 2022"),

        # Alien & Extraterrestrial Fabrications
        ("PROOF AT LAST: Flying saucer lands on White House lawn, alien ambassador demands pizza",
         "Tourists were stunned when a shimmering chrome flying disk touched down in the rose garden this morning. Extraterrestrial pilots emerged carrying signs demanding unlimited pepperoni slices or they will vaporize the planet!", "News", "February 04, 2022"),
        ("CLASSIFIED FILES: Millions of space aliens are living disguised as household pets across suburbs",
         "Veterinarians are breaking their non-disclosure agreements to warn pet owners. That weird behavior your cat exhibits is actually encrypted radio signals transmitting your credit card numbers back to Alpha Centauri!", "News", "April 25, 2022"),
        ("SECRET RADAR LEAK: Massive mothership the size of Texas parked directly behind the sun",
         "Astronomers who were fired for speaking out have published telescope images revealing a colossal metallic structure hiding in the sun's blind spot. Global elites are building underground bunkers as we speak!", "News", "June 30, 2022"),
        ("UNSEALED DOSSIER: Mars rover sends back photos of ancient alien shopping mall and parking lot",
         "NASA technicians smuggled out ultra-high-definition panoramic photos showing neon signs, escalators, and abandoned hover vehicles buried under Martian sand dunes. The truth can no longer be hidden!", "News", "August 16, 2022"),
        ("CONFIRMED: Government scientists have been communicating with dolphins about alien invasion plans",
         "Marine biologists equipped with underwater neural translation helmets discovered that oceanic marine mammals have formed an interstellar defense pact with friendly aquatic extraterrestrials!", "News", "October 22, 2022"),

        # Medical & Health Misinformation
        ("DOCTORS STUNNED: Rubbing raw garlic on your smartphone screen destroys 100% of bacterial infections",
         "Stop spending thousands on health insurance! A revolutionary home remedy proves that placing crushed cloves of garlic directly over your phone speaker creates a holistic shield against all seasonal viruses!", "News", "January 28, 2022"),
        ("ALERT: Synthetic plastic rice made of petroleum and potatoes flooding major grocery stores",
         "Unsuspecting families are eating toxic plastic grains designed to make people addicted to shopping malls. Perform the simple lighter test in your kitchen today before your next dinner!", "News", "March 15, 2022"),
        ("SECRET STUDY: Sleeping with onions in your socks regenerates lost teeth and repairs vision",
         "Dentists and optometrists are terrified of this suppressed peasant recipe from ancient Transylvania. Simply placing sliced onions inside wool socks overnight cures cavities and eliminates eyeglasses permanently!", "News", "May 14, 2022"),
        ("MASS PANIC: Tap water in major cities being infused with liquid mind control chemicals",
         "A brave laboratory whistleblower risked his life to share lab reports showing that municipal water towers have been poisoned with obedience serums that force citizens to pay higher sales taxes without complaining!", "politics", "July 24, 2022"),
        ("MIRACLE PROTOCOL: Breathing into a paper bag for three minutes eliminates all chronic diseases forever",
         "Medical universities tried to destroy the patent, but brave wellness influencers retrieved the secret technique. Big medicine is losing billions because oxygen recycling cures every ailment known to man!", "News", "September 18, 2022"),

        # Financial & Economic Sensationalism
        ("BANK RUN WARNING: Wall Street insiders dumping all assets before worldwide economic blackout tomorrow",
         "Anonymous billionaires have evacuated to remote Pacific islands after receiving urgent alerts about a scheduled complete global shutdown of the banking grid. Empty your accounts immediately while you still can!", "News", "February 18, 2022"),
        ("EXPOSED: The IRS is secretly minting trillion dollar chocolate coins to pay off the national debt",
         "An employee at the federal treasury smuggled out pictures of cocoa-flavored currency wrapped in gold foil. Economists are calling it the greatest financial fraud in modern civilization!", "politics", "April 02, 2022"),
        ("SHOCKING SCHEME: Credit card companies charging customers fees for looking at their balances online",
         "Greedy bank executives have activated hidden clauses that deduct ten dollars from your bank account every single time you log in to check your savings. Switch to hiding paper currency under your mattress now!", "News", "June 08, 2022"),
        ("ALERT: Automated teller machines around the nation begin dispensing Monopoly game paper money",
         "Furious customers across the country are posting videos of cash dispensers spitting out brightly colored board game money instead of real banknotes. Federal authorities claim it was just a printer calibration error!", "News", "August 28, 2022"),
        ("CONFIDENTIAL: Secret royal family buys all gold mines in South America using fraudulent IOUs",
         "Declassified shipping manifests expose how a renegade royal prince seized control of the world's precious metals market using fake promissory notes written on cocktail napkins!", "News", "October 19, 2022"),

        # Tech & Surveillance Paranoia
        ("TERRIFYING TRUTH: Smart refrigerators are recording your midnight snacking habits to raise your insurance",
         "Insurance agents are secretly buying bluetooth audio logs from kitchen appliances to deny health claims for anyone who eats cheese slices past midnight. Unplug your refrigerator before it destroys your life!", "News", "January 07, 2022"),
        ("LEAKED CODE: Popular video games contain hypnotic subliminal tones that force players to buy soda",
         "Game developers have confessed that high-pitched ultrasonic frequencies hidden in background soundtracks hypnotize teenagers into craving processed sugar and caffeinated beverages!", "News", "March 09, 2022"),
        ("ALARMING REPORT: GPS navigation apps intentionally directing traffic through haunted graveyards",
         "Motorists are reporting that mapping software suddenly recalculates routes to guide drivers down spooky abandoned roads inhabited by ghostly phantom hitchhikers. Tech executives are refusing to comment!", "News", "May 22, 2022"),
        ("HACKER WARNING: Your ceiling fan can be hacked by foreign spies to spin backwards and hypnotize you",
         "Cybersecurity experts reveal that smart ceiling fans connected to home wifi networks can be remotely modulated to emit hypnotic strobe patterns that make homeowners confess their deepest secrets!", "News", "July 12, 2022"),
        ("SHOCKING DISCOVERY: Wireless earbuds are beaming advertisements directly into your dreams at night",
         "Do you wake up suddenly craving brand name potato chips? Medical monitors confirm that blue tooth audio earbuds transmit audio commercials directly to the subconscious mind during REM sleep cycles!", "News", "September 29, 2022"),

        # Bizarre Nature & Animal Rumors
        ("MIRACLE NATURE: Flock of pigeons caught holding synchronized prayer circle in public park",
         "Hundreds of bewildered park visitors witnessed urban birds forming intricate geometric shapes and bowing their heads toward the heavens. Religious scholars claim it is an undeniable divine sign of impending judgment!", "News", "February 27, 2022"),
        ("PANIC IN ZOOS: Escaped gorilla learns to drive sports car and enters high-speed highway chase",
         "Police cruisers were left in the dust when an agile primate hot-wired a visitor's convertible and led state troopers on an eighty-mile pursuit while tuning the vehicle radio to classic rock hits!", "News", "April 19, 2022"),
        ("UNBELIEVABLE: Giant squirrel colony discovers secret subterranean gold mine under city library",
         "City maintenance workers stumbled into underground tunnels filled with millions of golden acorns carefully buried by mutant rodents. Town officials are plotting to seize the treasure to build a water park!", "News", "June 25, 2022"),
        ("SCANDAL: Scientists reveal that house cats have been secretly writing romance novels under pen names",
         "A prominent literary publisher confessed that several bestselling historical fiction novels were actually typed by Siamese cats walking back and forth across laptop keyboards late at night!", "News", "August 09, 2022"),
        ("WARNING: Wild raccoons organize union and demand paid vacation time from local garbage collectors",
         "Sanitation workers encountered picketing woodland creatures waving miniature protest signs demanding higher quality pizza crusts and rotten bananas in residential recycling bins across town!", "News", "October 31, 2022"),
    ]

    # Convert to DataFrames
    true_df = pd.DataFrame(true_articles, columns=["title", "text", "subject", "date"])
    fake_df = pd.DataFrame(fake_articles, columns=["title", "text", "subject", "date"])

    true_df.to_csv(true_csv_path, index=False)
    fake_df.to_csv(fake_csv_path, index=False)

    print(f"[INFO] Starter dataset generated successfully:")
    print(f"  - Genuine News (True.csv) : {len(true_df)} articles -> {true_csv_path}")
    print(f"  - Fake News (Fake.csv)    : {len(fake_df)} articles -> {fake_csv_path}")
    print("[INFO] If Kaggle Fake.csv and True.csv files are placed here, train_model.py will seamlessly train on the full corpus.")


if __name__ == "__main__":
    generate_sample_dataset()
