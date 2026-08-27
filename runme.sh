# pip install -r requirements.txt
# python STEP_02_UTSW_ImageTest_YP_T2_Clinic_Amp_v2.py \
#     --phase_num 5 \
#     --base_path /mnt/sda/Academics/Code/MyCode/UltraRecon-4D/26042101Foll25092901.Liver \
#     --MR_number 92441064  \
#     --st_date 20260410 \
#     >STEP_02_UTSW_ImageTest_YP_T2_Clinic_Amp_v2.log
python "4DMRI Synthesis_UTSW_DVFsmooth_YP_T2_Steps.py" \
    --base_path /mnt/sda/Academics/Code/MyCode/UltraRecon-4D/26042101Foll25092901.Liver \
    --MR_number 92441064 \
    --st_date 20260410 \
    --name_3d T2_AX_MVXD \
    --reference_file IM-301-0001.dcm \
    >Synthesis_UTSW_DVFsmooth_YP_T2_Steps.log
