# 数据集内容与格式说明（fb15k_custom）

本文件整理当前项目所用数据集 `data/fb15k_custom` 的文件内容与格式示例，并说明与官方 FB15k-237 的一致性。

## 1) 三元组文件（train/valid/test）

格式：`head \t relation \t tail`（三列，tab 分隔）

示例（各文件前 5 行）：

- `data/fb15k_custom/train.txt`
  ```
  /m/027rn	/location/country/form_of_government	/m/06cx9
  /m/017dcd	/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor	/m/06v8s0
  /m/07s9rl0	/media_common/netflix_genre/titles	/m/0170z3
  /m/01sl1q	/award/award_winner/awards_won./award/award_honor/award_winner	/m/044mz_
  /m/0cnk2q	/soccer/football_team/current_roster./sports/sports_team_roster/position	/m/02nzb8
  ```

- `data/fb15k_custom/valid.txt`
  ```
  /m/07pd_j	/film/film/genre	/m/02l7c8
  /m/06wxw	/location/location/time_zones	/m/02fqwt
  /m/01t94_1	/people/person/spouse_s./people/marriage/type_of_union	/m/04ztj
  /m/02xcb6n	/award/award_category/winners./award/award_honor/award_winner	/m/04x4s2
  /m/07f_7h	/film/film/release_date_s./film/film_regional_release_date/film_release_region	/m/04gzd
  ```

- `data/fb15k_custom/test.txt`
  ```
  /m/08966	/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month	/m/05lf_
  /m/01hww_	/music/performance_role/regular_performances./music/group_membership/group	/m/01q99h
  /m/09v3jyg	/film/film/release_date_s./film/film_regional_release_date/film_release_region	/m/0f8l9c
  /m/02jx1	/location/location/contains	/m/013t85
  /m/02jx1	/location/location/contains	/m/0m0bj
  ```

## 2) 实体/关系字典

格式：`id \t raw_id`

- `data/fb15k_custom/entities.dict`
  ```
  0	/m/010016
  1	/m/0100mt
  2	/m/0102t4
  3	/m/0104lr
  4	/m/0105y2
  ```

- `data/fb15k_custom/relations.dict`
  ```
  0	/american_football/football_team/current_roster./sports/sports_team_roster/position
  1	/award/award_category/category_of
  2	/award/award_category/disciplines_or_subjects
  3	/award/award_category/nominees./award/award_nomination/nominated_for
  4	/award/award_category/winners./award/award_honor/award_winner
  ```

## 3) 文本描述文件

格式：`id \t name [SEP] description`

- `data/fb15k_custom/entity2text.txt`
  ```
  /m/07gp9	Terminator_2:_Judgment_Day [SEP] Terminator 2: Judgment Day is a 1991 American science fiction action film...
  /m/027986c	Los_Angeles_Film_C’ritics_Association_Award_for_Best_Actor [SEP] The Los Angeles Film Critics Association Award for Best Actor...
  /m/079sf	Silver_Star [SEP] The Silver Star, officially the Silver Star Medal, is the United States third highest military decoration...
  /m/05vtw	Psychiatry [SEP] Psychiatry is the medical specialty devoted to the study, diagnosis, treatment, and prevention of mental disorders...
  /m/04999m	U.S._Lecce [SEP] Unione Sportiva Lecce or simply U.S. Lecce is an Italian football club...
  ```

- `data/fb15k_custom/relation2text.txt`
  ```
  /music/performance_role/regular_performances./music/group_membership/group	/music/performance_role/regular_performances./music/group_membership/group [SEP] music performance role regular performances. music group membership group
  /film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium	/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium [SEP] film film distributors. film film film distributor relationship film distribution medium
  /award/award_nominee/award_nominations./award/award_nomination/nominated_for	/award/award_nominee/award_nominations./award/award_nomination/nominated_for [SEP] award award nominee award nominations. award award nomination nominated for
  /olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics	/olympics/olympic_participating_country/athletes./olympics/olympic_athlete_affiliation/olympics [SEP] olympics olympic participating country athletes. olympics olympic athlete affiliation olympics
  /base/eating/practicer_of_diet/diet	/base/eating/practicer_of_diet/diet [SEP] base eating practicer of diet diet
  ```

说明：`entity2text.txt` 与 `relation2text.txt` 属于本项目自带的文本描述扩展，官方 FB15k-237 不包含此类文本文件。

## 4) 与官方 FB15k-237 一致性

当前数据规模与官方 FB15k-237 的常见统计一致：

- Entities: 14541  
- Relations: 237  
- Train: 272115  
- Valid: 17535  
- Test: 20466  

因此，`fb15k_custom` 在结构规模上与官方 FB15k-237 一致；文本描述文件是本项目额外提供的扩展信息。
