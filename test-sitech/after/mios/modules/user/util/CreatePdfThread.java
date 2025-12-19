package com.sitech.mios.modules.user.util;

import com.sitech.echd.infrastructure.topcache.util.SpringContextUtil;
import com.sitech.ijcf.message6.dt.out.OutDTO;
import com.sitech.mios.modules.common.bean.BusiResp;
import com.sitech.mios.modules.user.api.ITeAppInterTeapp;
import com.sitech.mios.modules.user.dao.beans.po.OperationLog;
import com.sitech.mios.modules.user.dao.mapper.PayMapper;
import com.sitech.mios.modules.user.service.impl.PayServiceImpl;
import com.sitech.miso.common.JsonUtils;
import com.sitech.miso.common.file.FileUtil;
import com.sitech.miso.common.file.MessageCode;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileItemFactory;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.commons.CommonsMultipartFile;
import org.springframework.mock.web.MockMultipartFile;
import org.apache.http.entity.ContentType;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by jincm on 2021/7/20.
 */
public class CreatePdfThread extends Thread{

    private final static Logger logger = LoggerFactory.getLogger(CreatePdfThread.class);

    private List<Map<String,Object>> billsList;

    private String serialNo;

    private String loginNo;

    public CreatePdfThread(List<Map<String, Object>> billsList, String serialNo, String loginNo) {

        this.billsList = billsList;
        this.serialNo = serialNo;
        this.loginNo = loginNo;
    }

    @Override
    public void run() {

        try {
            Thread.currentThread().setName("billBatchThread-"+serialNo);

            //生成文件并打包
            OperationLog log = new OperationLog();
            log.setId(serialNo);

            ITeAppInterTeapp teapp = ApplicationContextProvider.getBean(ITeAppInterTeapp.class);
            PayServiceImpl payService = ApplicationContextProvider.getBean(PayServiceImpl.class);

            payService.updateCreateFile(log);


            logger.info("批量文件上传成功");
        } catch (Exception e) {
            logger.error("批量文件生成失败",e);
        }finally {
            String filePath = String.format(".%spdf%sbatch%s%s%s", File.separatorChar,File.separatorChar,File.separatorChar,serialNo,File.separatorChar);
            FileUtil.deleteFromName(filePath);
        }

    }

   /* public MultipartFile fileToMultipartFile(File file) throws IOException{
        FileItem fileItem = createFileItem(file);
        MultipartFile multipartFile = new CommonsMultipartFile(fileItem);
        return multipartFile;
    }*/

    public MultipartFile fileToMultipartFile(File file) throws IOException {
    try (FileInputStream fileInputStream = new FileInputStream(file)) {
        return new MockMultipartFile(
                "file",
                file.getName(),
                ContentType.APPLICATION_OCTET_STREAM.toString(),
                fileInputStream
            );
        }
    }

    private static FileItem createFileItem(File file) throws IOException{
        FileItemFactory factory = new DiskFileItemFactory(16, null);
        FileItem item = factory.createItem("textField", "text/plain", true, file.getName());
        int bytesRead = 0;
        byte[] buffer = new byte[8192];

        return item;
    }

}
