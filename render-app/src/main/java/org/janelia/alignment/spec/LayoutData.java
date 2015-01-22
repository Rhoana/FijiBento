package org.janelia.alignment.spec;

import java.io.Serializable;

/**
 * Tile information from the layout file.
 *
 * @author Eric Trautman
 */
public class LayoutData implements Serializable {

    private final Integer sectionId;
    private final String temca;
    private final String camera;
    private final Integer imageRow;
    private final Integer imageCol;
    private final Double stageX;
    private final Double stageY;
    private final Double rotation;

    public LayoutData(final Integer sectionId,
                      final String temca,
                      final String camera,
                      final Integer imageRow,
                      final Integer imageCol,
                      final Double stageX,
                      final Double stageY,
                      final Double rotation) {
        this.sectionId = sectionId;
        this.temca = temca;
        this.camera = camera;
        this.imageRow = imageRow;
        this.imageCol = imageCol;
        this.stageX = stageX;
        this.stageY = stageY;
        this.rotation = rotation;
    }

    public Integer getSectionId() {
        return sectionId;
    }

    public String getTemca() {
        return temca;
    }

    public String getCamera() {
        return camera;
    }

    public Integer getImageRow() {
        return imageRow;
    }

    public Integer getImageCol() {
        return imageCol;
    }

    public Double getStageX() {
        return stageX;
    }

    public Double getStageY() {
        return stageY;
    }

    public Double getRotation() {
        return rotation;
    }

    /**
     * @return 32-bit "unique enough" tile id suitable for Karsh pipeline
     */
    public Integer getKarshTileId() {
        Integer karshTileId = null;
        if ((sectionId != null) && (imageCol != null) && (imageRow != null)) {
            karshTileId = (sectionId * 256 * 256) + (imageCol * 256) + imageRow;
        }
        return karshTileId;
    }

}