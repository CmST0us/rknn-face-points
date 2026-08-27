#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_RKNN_FACE_LANDMARK_COUNT 478
#define GST_RKNN_FACE_BLENDSHAPE_COUNT 52
#define GST_RKNN_FACE_META_API_TYPE (gst_rknn_face_meta_api_get_type())
#define GST_RKNN_FACE_META_INFO (gst_rknn_face_meta_get_info())

typedef struct _GstRknnFaceMeta {
  GstMeta meta;
  gboolean found;
  guint landmark_count;
  gfloat score;
  gfloat bbox[4];
  gfloat head_pose_deg[3];
  gfloat tongue_out;
  gfloat landmarks[GST_RKNN_FACE_LANDMARK_COUNT][3];
  gfloat blendshapes[GST_RKNN_FACE_BLENDSHAPE_COUNT];
  gdouble npu_ms;
} GstRknnFaceMeta;

GType gst_rknn_face_meta_api_get_type(void);
const GstMetaInfo* gst_rknn_face_meta_get_info(void);
GstRknnFaceMeta* gst_buffer_add_rknn_face_meta(GstBuffer* buffer);

static inline GstRknnFaceMeta* gst_buffer_get_rknn_face_meta(GstBuffer* buffer) {
  return (GstRknnFaceMeta*)gst_buffer_get_meta(buffer, GST_RKNN_FACE_META_API_TYPE);
}

G_END_DECLS
